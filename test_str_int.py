from lab4_proto import intToStr, strToInt

test_chars = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "_", "'"]
test_ints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]

# for i in range(len(test_chars)):
#     print(ord(test_chars[i]) - 95)

ret = strToInt(test_chars)

print(ret)

ret = intToStr(test_ints)
print(ret)