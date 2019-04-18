my_string = '0123456'
#length
print(len(my_string))
#find index of one character
print(my_string.find('6'))
#count() : occurence of substr
print(my_string.count('5'))
newstr = '123123123'
print(newstr.count('2'))
print(newstr.count('123'))

print('----------')
print('123' in my_string)
print('123' not in my_string)
is_in = '12' in my_string
print(is_in)

print('--------')
print(my_string.replace('0', 'X'))
print(my_string)

print('----在utf-8的编码下每个中文的长度是3-----')
print(len('你好'.encode('utf-8')))

print('----在gbk的编码下每个中文的长度是2-----')
print(len('你好'.encode('gbk')))

print('--------')
newstr = 'helllo'
print(newstr.replace('ll', '88'))

print('--------')
newstr = "hello world"
children = newstr.split(' ')
print(type(children))
print(children)