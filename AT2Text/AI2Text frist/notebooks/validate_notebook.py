#!/usr/bin/env python3
"""
验证 Colab notebook 的代码逻辑
解析 notebook JSON 并检查每个 cell 的代码
"""

import json
import sys
from pathlib import Path
import ast

def validate_notebook(notebook_path):
    """验证 notebook 文件"""
    import sys
    import io
    # 设置UTF-8输出
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("=" * 60)
    print(f"验证 Notebook: {notebook_path.name}")
    print("=" * 60)
    
    # 读取 notebook
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"❌ 无法读取 notebook: {e}")
        return False
    
    cells = nb.get('cells', [])
    print(f"\n找到 {len(cells)} 个 cells\n")
    
    issues = []
    python_cells = 0
    markdown_cells = 0
    
    for i, cell in enumerate(cells, 1):
        cell_type = cell.get('cell_type', 'unknown')
        source = ''.join(cell.get('source', []))
        
        if cell_type == 'markdown':
            markdown_cells += 1
            continue
        elif cell_type == 'code':
            python_cells += 1
            
            # 检查 Python 语法
            if source.strip():
                try:
                    # 尝试解析代码
                    ast.parse(source)
                    print(f"  [OK] Cell {i}: 语法正确 ({len(source)} 字符)")
                except SyntaxError as e:
                    error_msg = f"Cell {i}: 语法错误 - {e.msg} (line {e.lineno})"
                    print(f"  [ERROR] {error_msg}")
                    issues.append(error_msg)
                except Exception as e:
                    # 某些代码可能需要在运行时环境才能验证
                    print(f"  [WARN] Cell {i}: 需要运行时验证 ({type(e).__name__})")
        
        # 检查常见问题
        if 'BASE_DIR' in source and '=' in source:
            if '/content/drive/MyDrive' not in source:
                print(f"  [WARN] Cell {i}: BASE_DIR 可能需要更新路径")
        
        if 'subprocess.run' in source:
            if 'sys.executable' not in source and 'python' not in source.lower():
                print(f"  [WARN] Cell {i}: subprocess 调用可能需要指定 Python 路径")
    
    print(f"\n统计:")
    print(f"  - Markdown cells: {markdown_cells}")
    print(f"  - Python cells: {python_cells}")
    print(f"  - 问题数量: {len(issues)}")
    
    if issues:
        print(f"\n[ERROR] 发现 {len(issues)} 个问题:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print(f"\n[OK] Notebook 代码验证通过！")
        return True

def check_cell_dependencies():
    """检查 cell 之间的依赖关系"""
    print("\n" + "=" * 60)
    print("检查 Cell 依赖关系")
    print("=" * 60)
    
    notebook_path = Path(__file__).parent / 'colab_parquet_training.ipynb'
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"❌ 无法读取: {e}")
        return
    
    cells = nb.get('cells', [])
    defined_vars = set()
    required_vars = set()
    
    for i, cell in enumerate(cells, 1):
        if cell.get('cell_type') != 'code':
            continue
        
        source = ''.join(cell.get('source', []))
        
        # 检查定义的变量
        if 'QUICK_TEST' in source and '=' in source:
            defined_vars.add('QUICK_TEST')
        if 'BASE_DIR' in source and '=' in source:
            defined_vars.add('BASE_DIR')
        if 'datasets' in source and '=' in source:
            defined_vars.add('datasets')
        
        # 检查使用的变量
        if 'QUICK_TEST' in source and 'QUICK_TEST' not in defined_vars:
            if i > 1:  # 第一个cell可能定义它
                required_vars.add(('QUICK_TEST', i))
        if 'BASE_DIR' in source and 'BASE_DIR' not in defined_vars:
            if i > 1:
                required_vars.add(('BASE_DIR', i))
        if 'datasets' in source and 'datasets' not in defined_vars:
            if i > 1:
                required_vars.add(('datasets', i))
    
    if required_vars:
        print("[WARN] 可能的依赖问题:")
        for var, cell_num in required_vars:
            print(f"  - Cell {cell_num} 使用 {var}，但可能未定义")
    else:
        print("[OK] 未发现明显的依赖问题")

def main():
    """主函数"""
    notebook_path = Path(__file__).parent / 'colab_parquet_training.ipynb'
    
    if not notebook_path.exists():
        print(f"[ERROR] Notebook 不存在: {notebook_path}")
        return False
    
    # 验证 notebook
    success = validate_notebook(notebook_path)
    
    # 检查依赖
    check_cell_dependencies()
    
    print("\n" + "=" * 60)
    if success:
        print("[OK] Notebook 验证完成 - 代码看起来正常")
        print("   可以在 Colab 中运行")
    else:
        print("[WARN] Notebook 验证完成 - 发现一些问题")
        print("   请检查上述问题后再运行")
    print("=" * 60)
    
    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

