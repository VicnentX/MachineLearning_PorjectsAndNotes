# generate a empty bytes


b = bytes()
print(b)
b = b""
print(b)
b = bytes("hello world", encoding='GBK')
print(b)
s = b.decode(encoding='GBK')
print(s)


# string = b"XXXXX".decode()
# b = string.encode()   默认utf-8编码

