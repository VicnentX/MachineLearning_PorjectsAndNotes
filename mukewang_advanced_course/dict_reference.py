dict = {}

def can_palindrome(s, start, end, subs):
    cnt(s, start, end, set(), dict)
    print('dict size = ', len(dict))
    if subs >= dict[(start, end)]:
        return 1
    else:
        return 0


def cnt(s, start, end, set, dict):
    for i in range(start, end + 1):
        cur = s[i]
        if cur in set:
            set.remove(cur)
        else:
            set.add(cur)
        dict[(start, i)] = len(set) // 2


if __name__ == '__main__':
    print(can_palindrome('abcde', 1, 4, 1))

