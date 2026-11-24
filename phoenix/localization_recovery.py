"""Generic request schemas and recovery context for source-only localization."""
import copy
import json


def obj(properties, required=()):
    return {'type': 'object', 'properties': properties, 'required': list(required)}


def array(items):
    return {'type': 'array', 'items': items}


def enum(*values):
    return {'type': 'string', 'enum': list(values)}


def request_schema(name):
    text = {'type': 'string'}
    strings = array(text)
    line = {'type': 'integer', 'minimum': 1}
    evidence = array(obj({'file_path': text, 'start_line': line, 'end_line': line},
                         ('file_path', 'start_line', 'end_line')))
    if name == 'location_read':
        return obj({'operation': enum('read', 'search', 'list'), 'path': text,
                    'pattern': text, 'start_line': line, 'end_line': line,
                    'ignore_case': {'type': 'boolean'}}, ('path',))
    if name == 'location_checkpoint':
        return obj({
            'operation': enum('save', 'update'), 'notes': text,
            'requirements': array(obj({'id': text, 'text': text}, ('id', 'text'))),
            'hypotheses': array(obj({'id': text, 'idea': text, 'grounding': strings,
                                    'status': enum('open', 'retained', 'rejected'),
                                    'reason': text, 'rejection_evidence': strings}, ('id',))),
            'single_hypothesis_reason': text, 'active_hypothesis': text,
            'next_action': enum('continue', 'switch', 'finalize')}, ('notes',))
    coverage = obj({'requirement_id': text, 'status': enum('change', 'already_satisfied'),
                    'location_ids': strings, 'explanation': text, 'evidence': evidence},
                   ('requirement_id', 'status', 'location_ids', 'explanation'))
    review = obj({'area': enum('entrypoints', 'state_and_defaults', 'consumers_and_contracts',
                              'persistence_and_serialization', 'dependencies_and_registration',
                              'compatibility_and_errors'),
                  'status': enum('change', 'unchanged', 'not_applicable'),
                  'location_ids': strings, 'explanation': text, 'evidence': evidence},
                 ('area', 'status', 'location_ids', 'explanation'))
    modification = obj({'id': text, 'reason': text,
                        'location': obj({'start_line': line, 'end_line': line}, ('start_line', 'end_line'))},
                       ('id', 'reason'))
    candidate = obj({'id': text, 'hypothesis_id': text, 'implementation_outline': text,
                     'files_to_modify': array(obj({'file_path': text, 'operation': enum('modify', 'create'),
                                                  'modifications': array(modification)},
                                                 ('file_path', 'modifications'))),
                     'requirement_coverage': array(coverage), 'dependency_review': array(review)},
                    ('id', 'hypothesis_id', 'implementation_outline', 'files_to_modify',
                     'requirement_coverage', 'dependency_review'))
    return obj({'operation': enum('save', 'get', 'list', 'remove', 'validate'),
                'candidate': candidate, 'id': text, 'reason': text, 'rejection_evidence': strings})


def repeated_action_message(last_action, repeated, limit):
    """Warn about successful-but-stalled actions as well as rejected calls."""
    if not last_action or repeated < 2:
        return ''
    return (
        f'\nREPEATED ACTION WARNING: The same tool and arguments were used {repeated} '
        'consecutive times. Repeating an unchanged read/search against this fixed checkout '
        'does not produce new evidence; a successful tool call does not mean investigation progressed. '
        'Do not repeat that action again. Identify the missing fact and change the search '
        'pattern or scope, read a relevant caller/declaration, or investigate another open branch '
        'when the currently allowed tools permit it. If the evidence is unavailable, record '
        'the unresolved assumption honestly. Preserve existing requirements and drafts; do not '
        'invent evidence or mark an incomplete plan complete. Follow any required checkpoint '
        'before changing branches or saving. '
        + (f'The unchanged hard stop is {limit} consecutive identical calls. ' if limit > 0 else '')
        + 'Changing arguments only cosmetically is not progress.\n')


def recovery_message(error, data, repeated):
    inv = data.get('investigation', {})
    state = {'requirements': inv.get('requirements', []),
             'active_hypothesis': inv.get('active_hypothesis'),
             'hypotheses': [{k: h[k] for k in ('id', 'status', 'candidate_id') if k in h}
                            for h in inv.get('hypotheses', [])],
             'single_hypothesis_reason': inv.get('single_hypothesis_reason'),
             'saved_candidates': [c['id'] for c in data.get('candidates', [])]}
    return ('\nLAST TOOL REJECTION (the rejected operation did not update saved state):\n' + error +
            '\nCorrect the named field before retrying. The OUTER arguments must be {"request": {...}}; never flatten that wrapper. '
            'Routine checkpoints need notes and only changed fields; omit unchanged requirements/hypotheses. '
            'Do not rewrite registered requirement text. If there is only one grounded hypothesis, supply '
            'your own single_hypothesis_reason describing the material decisions examined; '
            'explicit interfaces alone do not prove uniqueness. Do not invent an alternative. '
            'Before saving another branch, checkpoint active_hypothesis to that registered open id. '
            'Do not relabel unresolved behavior as satisfied: inspect source, identify necessary edits and '
            'complete coverage, or leave the hypothesis open. unchanged/already_satisfied claims need actual '
            'source ranges. If no complete plan is supported, abort honestly; never fabricate evidence.\n'
            f'Consecutive identical calls: {repeated}. Repeating unchanged arguments cannot fix this error.\n'
            'AUTHORITATIVE SAVED STATE (rejected proposals are not saved):\n' +
            json.dumps(state, ensure_ascii=False))


def compact_rejected_turns(messages, rejected_ids, count_tokens, *, target_tokens, keep_last=4):
    """Elide old rejected draft turns in API copies; retain all accepted turns.

    The most recent turns (including the latest failed proposal) remain intact.
    Full responses and observations remain in trajectory.jsonl. Never discard an
    unmatched tool call or any round containing a successful tool operation.
    """
    groups = []
    for i, m in enumerate(messages):
        calls = m.get('tool_calls', []) if m.get('role') == 'assistant' else []
        if calls:
            groups.append((i, calls))
    result = copy.deepcopy(messages)
    removed = set()
    count = count_tokens(result)
    omitted = 0
    for i, calls in groups[:-keep_last] if keep_last else groups:
        if count <= target_tokens:
            break
        ids = [c['id'] for c in calls]
        if not set(ids) <= rejected_ids:
            continue
        end = i + 1
        while end < len(messages) and messages[end].get('role') == 'tool':
            end += 1
        if sorted(m['tool_call_id'] for m in messages[i + 1:end]) != sorted(ids):
            continue
        # No invented tool arguments or results. A clear user note replaces the
        # entire failed exchange, leaving the successful protocol history intact.
        errors = [m.get('content', '')[:1000] for m in messages[i + 1:end]]
        replacement = {'role': 'user', 'content':
                     '[Context compaction: an older rejected tool exchange was omitted. '
                     'It made no saved-state change; full draft is retained in trajectory.jsonl.]\n' +
                     json.dumps({'tools': [c['function']['name'] for c in calls], 'errors': errors})}
        previous = result[i]
        result[i] = replacement
        proposed_removed = removed | set(range(i + 1, end))
        after = count_tokens([m for j, m in enumerate(result) if j not in proposed_removed])
        if after >= count:
            result[i] = previous
            continue
        removed = proposed_removed
        omitted += 1
        count = after
    return [m for j, m in enumerate(result) if j not in removed], count, omitted


def normalize_tool_arguments(name, arguments):
    """Repair only an unambiguous missing envelope, never plan contents."""
    args = json.loads(arguments) if isinstance(arguments, str) else copy.deepcopy(arguments)
    if not isinstance(args, dict):
        raise ValueError('Tool arguments must be a JSON object')
    if name not in ('location_read', 'location_checkpoint', 'location_candidate'):
        return args, []
    if 'request' in args:
        if set(args) != {'request'}:
            raise ValueError('Use one outer request object; do not mix wrapped and top-level fields')
        inner = args['request']
        if isinstance(inner, str):
            # Existing decode_request handles the legacy string representation.
            return args, []
        repaired = []
    else:
        inner = args
        repaired = ['wrapped_missing_request']
    if not isinstance(inner, dict):
        raise ValueError('The request field must contain a JSON object')
    if name == 'location_checkpoint' and 'candidate' in inner:
        raise ValueError('A checkpoint cannot save a candidate. First call location_checkpoint with '
                         '{"request":{"notes":"summarize current source findings","next_action":"continue"}}. '
                         'After that succeeds, call location_candidate with {"request":{"operation":"save",'
                         '"candidate": YOUR_EXISTING_PLAN}}. Your rejected draft remains in the trajectory; '
                         'it has not been published or validated.')
    schema = request_schema(name)
    unknown = set(inner) - set(schema['properties'])
    missing = set(schema['required']) - set(inner)
    if unknown or missing:
        raise ValueError(f'{name} expects {{"request": {{...}}}}; '
                         f'missing fields: {sorted(missing)}; unexpected fields: {sorted(unknown)}. '
                         'Fix only these fields; preserve the existing evidence and plan.')
    return {'request': inner}, repaired
