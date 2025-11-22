"""
检查token ID 4对应什么字符
"""
import sys
import io

# 设置UTF-8编码
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append('.')

from preprocessing.text_cleaning import Tokenizer

# 初始化tokenizer
tokenizer = Tokenizer()

print("\n" + "="*60)
print("检查Token ID 4")
print("="*60 + "\n")

token_id = 4

# 检查token ID 4对应的字符
if token_id in tokenizer.idx_to_char:
    char = tokenizer.idx_to_char[token_id]
    print(f"Token ID {token_id} 对应的字符: '{char}'")
    
    # 检查是否是特殊token
    special_tokens = {'<pad>', '<unk>', '<sos>', '<eos>', '<blank>'}
    if char in special_tokens:
        print(f"  这是一个特殊token: {char}")
    else:
        print(f"  这是一个普通字符")
else:
    print(f"Token ID {token_id} 不在vocab中")

# 显示vocab的前20个token
print(f"\nVocab大小: {len(tokenizer)}")
print(f"\n前20个token:")
for i in range(min(20, len(tokenizer))):
    char = tokenizer.idx_to_char.get(i, '<unk>')
    print(f"  ID {i}: '{char}'")

# 检查token ID 4在vocab中的位置
print(f"\nToken ID 4的详细信息:")
if 4 in tokenizer.idx_to_char:
    char = tokenizer.idx_to_char[4]
    print(f"  字符: '{char}'")
    print(f"  字符代码: {ord(char) if char else 'N/A'}")
    print(f"  是否空白: {char.isspace() if char else False}")
    print(f"  是否特殊token: {char in {'<pad>', '<unk>', '<sos>', '<eos>', '<blank>'}}")

print("\n" + "="*60)

