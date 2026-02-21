#!/usr/bin/env python3
import os
import json
from pathlib import Path


def find_patch_files_in_final_folders(root_path):
    result = {}
    root = Path(root_path)
    final_folders = [folder for folder in root.iterdir() if folder.is_dir() and folder.name.endswith('-final')]
    print(f"找到 {len(final_folders)} 个 -final 文件夹:")
    for folder in final_folders:
        print(f"处理文件夹: {folder.name}")
        patch_files = list(folder.glob('*.patch'))
        if patch_files:
            patch_file = patch_files[0]
            try:
                with open(patch_file, 'r', encoding='utf-8') as f:
                    patch_content = f.read()
                original_id = folder.name.replace('-sample-final', '')
                result[original_id] = {
                    "model_name_or_path": "default__openai--GLM-4.7__t-0.70__p-1.00__c-0.00___swe_bench_verified_test",
                    "instance_id": original_id,
                    "model_patch": patch_content
                }
                print(f"✓ 成功读取 {patch_file.name}")
            except Exception as e:
                print(f"✗ 读取文件失败 {patch_file}: {e}")
        else:
            print(f"未找到 .patch 文件")
    return result

def main():
    root_directory = ""  # Please Enter The absolute path obtained in Phase 2
    patches_data_final = find_patch_files_in_final_folders(root_directory)
    if patches_data:
        output_file = os.path.join(root_directory, "answer_final.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(patches_data, f, indent=4, ensure_ascii=False)
        print(f"\n✓ 成功生成 {output_file}")
        print(f"共处理了 {len(patches_data)} 个patch文件")
    else:
        print("未找到任何patch文件")

if __name__ == "__main__":
    main()
