#!/usr/bin/env python3
"""
测试脚本：验证 Colab notebook 的关键功能
可以在本地运行来检查代码是否有问题
"""

import os
import sys
from pathlib import Path

def test_imports():
    """测试所有必需的导入"""
    print("=" * 60)
    print("测试 1: 检查 Python 导入")
    print("=" * 60)
    
    required_packages = [
        'torch', 'torchaudio', 'transformers', 'librosa', 
        'soundfile', 'numpy', 'pandas', 'yaml', 'datasets'
    ]
    
    failed = []
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError as e:
            print(f"  ❌ {pkg}: {e}")
            failed.append(pkg)
    
    if failed:
        print(f"\n⚠️ {len(failed)} 个包导入失败")
        return False
    else:
        print("\n✅ 所有包导入成功")
        return True

def test_cuda():
    """测试 CUDA 可用性"""
    print("\n" + "=" * 60)
    print("测试 2: 检查 CUDA/GPU")
    print("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA 可用")
            print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ CUDA 版本: {torch.version.cuda}")
            return True
        else:
            print("  ⚠️ CUDA 不可用（将使用 CPU）")
            return False
    except ImportError:
        print("  ❌ PyTorch 未安装")
        return False

def test_project_structure():
    """测试项目结构"""
    print("\n" + "=" * 60)
    print("测试 3: 检查项目结构")
    print("=" * 60)
    
    required_dirs = [
        'training', 'models', 'configs', 'database', 
        'preprocessing', 'scripts'
    ]
    
    base_path = Path(__file__).parent.parent
    missing = []
    
    for dir_name in required_dirs:
        dir_path = base_path / dir_name
        if dir_path.exists():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ 不存在")
            missing.append(dir_name)
    
    if missing:
        print(f"\n⚠️ {len(missing)} 个目录缺失")
        return False
    else:
        print("\n✅ 项目结构完整")
        return True

def test_config_files():
    """测试配置文件"""
    print("\n" + "=" * 60)
    print("测试 4: 检查配置文件")
    print("=" * 60)
    
    base_path = Path(__file__).parent.parent
    config_path = base_path / 'configs' / 'default.yaml'
    
    if config_path.exists():
        print(f"  ✓ configs/default.yaml 存在")
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            print(f"  ✓ 配置文件可解析")
            print(f"  - batch_size: {cfg.get('batch_size', 'N/A')}")
            print(f"  - num_epochs: {cfg.get('num_epochs', 'N/A')}")
            return True
        except Exception as e:
            print(f"  ❌ 配置文件解析失败: {e}")
            return False
    else:
        print(f"  ❌ configs/default.yaml 不存在")
        return False

def test_training_script():
    """测试训练脚本"""
    print("\n" + "=" * 60)
    print("测试 5: 检查训练脚本")
    print("=" * 60)
    
    base_path = Path(__file__).parent.parent
    train_script = base_path / 'training' / 'train.py'
    
    if train_script.exists():
        print(f"  ✓ training/train.py 存在")
        # 检查是否可以导入
        try:
            sys.path.insert(0, str(base_path))
            # 不实际导入，只检查语法
            with open(train_script, 'r', encoding='utf-8') as f:
                code = f.read()
            compile(code, str(train_script), 'exec')
            print(f"  ✓ 训练脚本语法正确")
            return True
        except SyntaxError as e:
            print(f"  ❌ 语法错误: {e}")
            return False
    else:
        print(f"  ❌ training/train.py 不存在")
        return False

def test_parquet_loading():
    """测试 parquet 文件加载逻辑"""
    print("\n" + "=" * 60)
    print("测试 6: 检查 Parquet 加载逻辑")
    print("=" * 60)
    
    try:
        from datasets import load_dataset, Audio
        print("  ✓ datasets 库可用")
        print("  ✓ Audio 功能可用")
        print("  ℹ️  注意：需要实际的 parquet 文件才能完整测试")
        return True
    except ImportError as e:
        print(f"  ❌ datasets 库导入失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("🧪 Colab Notebook 设置测试")
    print("=" * 60)
    print("\n这个脚本会检查 notebook 运行所需的关键组件")
    print("在 Colab 中运行 notebook 之前，可以先运行此脚本验证\n")
    
    results = []
    results.append(("导入检查", test_imports()))
    results.append(("CUDA/GPU", test_cuda()))
    results.append(("项目结构", test_project_structure()))
    results.append(("配置文件", test_config_files()))
    results.append(("训练脚本", test_training_script()))
    results.append(("Parquet 加载", test_parquet_loading()))
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！Notebook 应该可以正常运行")
    else:
        print(f"\n⚠️ {total - passed} 个测试失败，请检查上述问题")
        print("   修复这些问题后再在 Colab 中运行 notebook")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

