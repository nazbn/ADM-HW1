#----------------------------- Numpy > Arrays ----------------------------------

import numpy

def arrays(arr):
    a = numpy.array(arr,float)
    return numpy.flip(a)

arr = input().strip().split(' ')
result = arrays(arr)
print(result)

#----------------------------- Numpy > Shape and Reshape -----------------------

import numpy

arr = numpy.array(list(map(int, input().split())))
print(numpy.reshape(arr, (3, 3)))

#----------------------------- Numpy > Transpose and Flatten -------------------

import numpy

n, m = map(int, input().split())
lst = []
for _ in range(n):
    lst.append(list(map(int, input().split())))
arr = numpy.array(lst)
print(numpy.transpose(arr))
print(arr.flatten())

#----------------------------- Numpy > Concatenate -----------------------------

import numpy

n, m, p = map(int, input().split())
lst = []
for _ in range(n):
    lst.append(list(map(int, input().split())))
a = numpy.array(lst)
lst = []
for _ in range(m):
    lst.append(list(map(int, input().split())))
b = numpy.array(lst)
print(numpy.concatenate((a, b), axis = 0))

#----------------------------- Numpy > Zeros and Ones --------------------------

import numpy

axis = tuple(map(int, input().split()))
print(numpy.zeros(axis, dtype = numpy.int))
print(numpy.ones(axis, dtype = numpy.int))

#----------------------------- Numpy > Eye and Identity ------------------------

import numpy

n, m = map(int, input().split())
a = numpy.eye(n, m)
print(str(a).replace('1',' 1').replace('0',' 0')) #Otherwise the website wouldn't accept it!

#----------------------------- Numpy > Array Mathematics -----------------------

import numpy

n, m = map(int, input().split())
lst = []
for _ in range(n):
    lst.append(list(map(int, input().split())))
a = numpy.array(lst)
lst = []
for _ in range(n):
    lst.append(list(map(int, input().split())))
b = numpy.array(lst)
print(a+b)
print(a-b)
print(a*b)
print(a//b)
print(a%b)
print(a**b)

#----------------------------- Numpy > Floor, Ceil and Rint --------------------

import numpy

numpy.set_printoptions(sign=' ') #Just for website's sake
a = numpy.array(list(map(float, input().split())))
print(numpy.floor(a))
print(numpy.ceil(a))
print(numpy.rint(a))

#----------------------------- Numpy > Sum and Prod ----------------------------

import numpy

n, m = map(int, input().split())
lst = []
for _ in range(n):
    lst.append(list(map(int, input().split())))
arr = numpy.array(lst)
print(numpy.prod(numpy.sum(arr, axis = 0)))

#----------------------------- Numpy > Min and Max -----------------------------

import numpy

n, m = map(int, input().split())
lst = []
for _ in range(n):
    lst.append(list(map(int, input().split())))
arr = numpy.array(lst)
print(numpy.max(numpy.min(arr, axis = 1)))

#----------------------------- Numpy > Mean, Var, and Std ----------------------

import numpy

numpy.set_printoptions(legacy='1.13') #Same here
n, m = map(int, input().split())
lst = []
for _ in range(n):
    lst.append(list(map(int, input().split())))
arr = numpy.array(lst)
print(numpy.mean(arr, axis = 1))
print(numpy.var(arr, axis = 0))
print(numpy.std(arr))

#----------------------------- Numpy > Dot and Cross ---------------------------

import numpy

n = int(input())
lst = []
for _ in range(n):
    lst.append(list(map(int, input().split())))
a = numpy.array(lst)
lst = []
for _ in range(n):
    lst.append(list(map(int, input().split())))
b = numpy.array(lst)
print(numpy.dot(a, b))

#----------------------------- Numpy > Inner and Outer -------------------------

import numpy

a = numpy.array(list(map(int, input().split())))
b = numpy.array(list(map(int, input().split())))
print(numpy.inner(a, b))
print(numpy.outer(a, b))

#----------------------------- Numpy > Polynomials -----------------------------

import numpy

coe = list(map(float, input().split()))
print(numpy.polyval(coe, int(input())))

#----------------------------- Numpy > Linear Algebra --------------------------

import numpy

n = int(input())
lst = []
for _ in range(n):
    lst.append(list(map(float, input().split())))
arr = numpy.array(lst)
print(round(numpy.linalg.det(arr), 2))

#----------------------------- Standardize Mobile Number Using Decorators ------

import re

def wrapper(f):
    def fun(l):
        # complete the function
        for i in range(len(l)):
            m = re.match(r"(0|91|\+91)?(\d{5})(\d{5})", l[i])
            l[i] = "+91 "+ m.groups()[1] + " " + m.groups()[2]
        ret = f(l)
        return ret
    return fun

#----------------------------- Decorators 2 - Name Directory -------------------

import operator

def person_lister(f):
    def inner(people):
        for i in range(len(people)):
            people[i][2] = int(people[i][2])
        people.sort(key=lambda x: x[2])
        for i in range(len(people)):
            people[i] = f(people[i])
        return(iter(people))
    return inner

@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]

if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')

#--------------------------- Birthday Cake Candles -----------------------------

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the birthdayCakeCandles function below.
def birthdayCakeCandles(ar):
    m = max(ar)
    c = 0
    for element in ar:
        if element == m:
            c += 1
    return c

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    ar_count = int(input())

    ar = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(ar)

    fptr.write(str(result) + '\n')

    fptr.close()

#-------------------------- Kangaroo -------------------------------------------

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the kangaroo function below.
def kangaroo(x1, v1, x2, v2):
    if v1 == v2:
        if x1 == x2:
            return "YES"
        else:
            return "NO"
    t = (x2 - x1)/(v1 - v2)
    if t >= 0 and t == int(t) :
        return "YES"
    else:
        return "NO"

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    x1V1X2V2 = input().split()

    x1 = int(x1V1X2V2[0])

    v1 = int(x1V1X2V2[1])

    x2 = int(x1V1X2V2[2])

    v2 = int(x1V1X2V2[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

#-------------------------- Viral/Strange Advertising --------------------------

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the viralAdvertising function below.
def viralAdvertising(n):
    shared = 5
    likes = 0
    cumulative = 0
    for i in range(n):
        likes = shared//2
        cumulative += likes
        shared = likes*3
    return cumulative


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()

#------------------------- Recursive Digit Sum ---------------------------------

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the superDigit function below.
def superDigit(n, k):
    if len(n)*k == 1:
        return int(n)
    else:
        sums = 0
        for i in range(len(n)):
            sums += int(n[i])
        return superDigit(str(sums*k), 1)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = nk[0]

    k = int(nk[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()

#------------------------- Insertion Sort - Part 1 -----------------------------

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the insertionSort1 function below.
def insertionSort1(n, arr):
    n -= 1
    x = arr[n]
    while n > 0 and x < arr[n-1]:
        arr[n] = arr[n-1]
        n -= 1
        print(*arr)
    arr[n] = x
    print(*arr, end=" ")


if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)

#------------------------ Insertion Sort - Part 2 ------------------------------

#!/bin/python3

import math
import os
import random
import re
import sys

def insertionSort1(n1, arr):
    
    n1 -= 1
    x = arr[n1]
    while n1 > 0 and x < arr[n1-1]:
        arr[n1] = arr[n1-1]
        n1 -= 1
    arr[n1] = x
    print(*arr)
    return arr
# Complete the insertionSort2 function below.
def insertionSort2(n, arr2):
    #print(arr)
    if n == 1:
        return arr2
    else:
        #print("before call : " , []+[arr2[n-1]])
        return insertionSort1(n, insertionSort2(n-1, arr2))# + [arr2[n-1]])

if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)

#----------------------------- Exceptions --------------------------------------

t = int(input())
for i in range(t):
    a, b = input().split()
    try:
        print(int(a)//int(b))
    except Exception as e:
        print("Error Code:",e)

#---------------------------- XML 1 - Find the Score ---------------------------

import sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
    s = 0
    for child in node:
        s += get_attr_number(child)
    s += len(node.attrib)
    return s

if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))

#-------------------------- XML2 - Find the Maximum Depth ----------------------

import xml.etree.ElementTree as etree

maxdepth = 0
def depth(elem, level):
    global maxdepth
    level += 1
    for child in elem:
        depth(child, level)
    if level > maxdepth:
        maxdepth = level

if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml =  xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)

#----------------- Regex > Detect Floating Point Number ------------------------

import re
 
for i in range(int(input())):
    print(bool(re.match(r"\A[+-0123456789]\d*\.\d+\Z", input())))

#----------------- Regex > Re.split() ------------------------------------------

regex_pattern = r"\W"	# Do not delete 'r'.
import re
print("\n".join(re.split(regex_pattern, input())))

#----------------- Regex > Group(), Groups() & Groupdict() ---------------------

import re

m = re.match(r'.*?([a-zA-Z0-9])\1', input())
if m == None:
    print(-1)
else:
    print(m.group(1))

#----------------- Regex > Re.findall() & Re.finditer() ------------------------

import re
lst = re.findall(r"(?<=[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm])([AEIOUaeiou][AEIOUaeiou]+)(?=[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm])", input())
if lst == []:
    print(-1)
else:
    for item in lst:
        print(item)

#----------------- Regex > Re.start() & Re.end() -------------------------------

import re

s = input()
k = input()
match_objects = re.finditer(r''+ k[0] + '(?='+ k[1:] +')',s)
lst = list(map(lambda x: x.start(),match_objects))
if lst == []:
    print((-1, -1))
else:
    for item in lst:
        print((item, item + len(k) -1))

#----------------- Regex > Regex Substitution ----------------------------------

import re

n = int(input())
for _ in range(n):
    s = input()
    s = re.sub(r"(?<=\s)\|\|(?=\s)", "or" , s)
    s = re.sub(r"(?<=\s)&&(?=\s)", "and" , s)
    print(s)

#----------------- Regex > Validating Roman Numerals ---------------------------

regex_pattern = r"^(I(?=X))?(X(?=C))?(C(?=M))?M{0,3}(I(?=X))?(X(?=C))?(C(?=M))?((?<=C|X|I)M)?(I(?=X))?(X(?=C))?(C(?=D))?D?(I(?=X))?(X(?=C))?C{0,3}(I(?=X))?(X(?=L))?L?(I(?=X))?X{0,3}(I(?=V))?V?I{0,3}$"	# Do not delete 'r'.

import re
print(str(bool(re.match(regex_pattern, input()))))
    # Only one I, X, and C can be used as the leading numeral in part of a subtractive pair.
    # I can only be placed before V and X.
    # X can only be placed before L and C.
    # C can only be placed before D and M.

    # Five in a row of any digit is not allowed
    # Some digits are allowed in runs of up to 4. They are I,X,C, and M. The others (V,L,D) can only appear singly.
    # Some lower digits can come before a higher digit, but only if they appear singly. E.g. "IX" is ok but "IIIX" is not.
    # But this is only for pairs of digits. Three ascending numbers in a row is invalid. E.g. "IX" is ok but "IXC" is not.
    # A single digit with no runs is always allowed

#  I can be placed before V (5) and X (10) to make 4 and 9.
#   ⋅ X can be placed before L (50) and C (100) to make 40 and 90.
#   ⋅ C can be placed before D (500) and M (1000) to make 400 and 900.

#----------------- Regex > Validating phone numbers ----------------------------

import re

for _ in range(int(input())):
    m = re.match("^[789]\d{9}$", input())
    if bool(m) == True:
        print("YES")
    else:
        print("NO")

#----------------- Regex > Validating and Parsing Email Addresses --------------

import email.utils
import re

for _ in range(int(input())):
    t = email.utils.parseaddr(input())
    if bool(re.match(r"^[a-zA-Z](\w|-|\.)+@[a-zA-Z]+\.[a-zA-Z]{1,3}$", t[1])) == True:
        print(email.utils.formataddr(t))

#----------------- Regex > Hex Color Code --------------------------------------

import re

for _ in range(int(input())):
    m = re.finditer(r"(?<!^)(#[a-fA-F0-9]{6}|#[a-fA-F0-9]{3})", input().lstrip())
    for item in list(map(lambda x: x.group(1), m)):
        print(item)

#------------------------- Regex > HTML Parser - Part 1 ------------------------

from html.parser import HTMLParser

# create a subclass and override the handler methods
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Start :", tag)
        for attr in attrs:
            print("-> " + str(attr[0]) + " > " + str(attr[1]))
    def handle_endtag(self, tag):
        print("End   :", tag)
    def handle_startendtag(self, tag, attrs):
        print("Empty :", tag)
        for attr in attrs:
            print("-> " + str(attr[0]) + " > " + str(attr[1]))

s = ""
for _ in range(int(input())):
    s = s + input()
parser = MyHTMLParser()
parser.feed(s)
parser.close()

#------------------------- Regex > HTML Parser - Part 2 ------------------------

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_data(self, data):
        if data != "\n":
            print(">>> Data")
            print(data)
    def handle_comment(self, data):
        if data.find("\n") == -1:
            print(">>> Single-line Comment")
        else:
            print(">>> Multi-line Comment")
        print(data)
  
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()

#----------------- Regex > Detect HTML Tags, Attributes and Attribute Values ---

import re

s = ""
for _ in range(int(input())):
    s += input()
s = re.sub(r"<!--.*?-->", "", s)
m = re.findall(r"<.+?>|</.+?>|<.+?/>", s)
for item in m:
    if re.match(r"</.+>", item): #---------------------- End
        continue
    elif re.match(r"<.+/>", item): #-------------------Empty
        m = re.search(r"^<\s*?(\S*)", item)
        print(m.group(1)) #--- Print the tag and then remove it
        #print(item[m.end():-2])
        f = re.finditer(r"(\S+)\s*=\s*\"(.*?)\"|(\S+)\s*=\s*\'(.*?)\'|(\S+)", item[m.end():-2])
        if f != None:
            lst = list(map(lambda x : x.groups(), f))
            #print(lst)
            for element in lst:
                if element[0] == None and element[2] == None: #---- No Value
                    print("-> " + element[4] + " > None")
                elif element[0] == None and element[4] == None: #--- With ''
                    print("-> " + element[2] + " > " + element[3])
                elif element[2] == None and element[4] == None: #--- With ""
                    print("-> " + element[0] + " > " + element[1])
    elif re.match(r"<.+>", item): #---------------------Start
        m = re.search(r"^<\s*?(\S*)", item)
        if m.end() == len(item): #--- Print the tag and then remove it
            print(m.group(1)[:-1]) 
        else:
            print(m.group(1)) #---- if it has attrs ">" will be at the end
        f = re.finditer(r"(\S+)\s*=\s*\"(.*?)\"|(\S+)\s*=\s*\'(.*?)\'|(\S+)", item[m.end():-1])
        if f != None:
            lst = list(map(lambda x : x.groups(), f))
            #print(lst)
            for element in lst:
                if element[0] == None and element[2] == None: #--- No Value
                    print("-> " + element[4] + " > None")
                elif element[0] == None and element[4] == None: #--- With ''
                    print("-> " + element[2] + " > " + element[3])
                elif element[2] == None and element[4] == None: #--- With ""
                    print("-> " + element[0] + " > " + element[1])

#------------------------- Regex > Validating UID ------------------------------

import re

for _ in range(int(input())):
    s = input()
    flag = True
    if re.match(r"^[a-zA-Z0-9]{10}$", s) == None:
        flag = False
    m = re.findall(r"([\w*]).*\1", s)
    if m != []:
        flag = False
    m = re.findall(r"[A-Z]", s)
    if len(m) < 2:
        flag = False
    m = re.findall(r"[0-9]", s)
    if len(m) < 3:
        flag = False

    if flag == True:
        print("Valid")
    else:
        print("Invalid")

#------------------------ Regex > Validating Credit Card Numbers ---------------

import re

for _ in range(int(input())):
    c = input()
    if re.match(r"^[456]\d{3}-?\d{4}-?\d{4}-?\d{4}$", c) and re.findall(r"(\d)-?\1-?\1-?\1", c) == []:
        print("Valid")
    else:
        print("Invalid")

#------------------------ Regex > Validating Postal Codes ----------------------

regex_integer_in_range = r"^[1-9]\d{5}$"	# Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"(\d)(?=.\1)"	# Do not delete 'r'.


import re
P = input()

print (bool(re.match(regex_integer_in_range, P)) 
and len(re.findall(regex_alternating_repetitive_digit_pair, P)) < 2)

#------------------------ Regex > Matrix Script --------------------------------

#!/bin/python3

import math
import os
import random
import re
import sys

first_multiple_input = input().rstrip().split()
n = int(first_multiple_input[0])
m = int(first_multiple_input[1])
matrix = []
for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)
columns = zip(*matrix)
l = []
for col in columns:
    l += list(col)
s = ''.join(l)
print(re.sub(r"(?<=[a-zA-Z0-9])([!@#$%&]|\s)+(?=[a-zA-Z0-9])", " ", s))

#------------------------- Map and Lambda Function -----------------------------

cube = lambda x: x**3 # complete the lambda function 

def fibonacci(n):
    # return a list of fibonacci numbers
    l = []
    for i in range(n):
        if i == 0 :
            l.append(0)
        elif i == 1:
            l.append(1)
        else:
            l.append(l[i-1]+l[i-2])
    return l

if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))

#------------------------- Zipped! ---------------------------------------------

import statistics
n, x = map(int, input().split())
mark_sheet = []
for i in range(x):
    lst = list(map(float, input().split()))
    mark_sheet.append(lst)
zipped = zip(*mark_sheet)
for item in zipped:
    print(statistics.mean(item))
    
#------------------------- Athlete Sort ---------------------------------------

#!/bin/python3

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())
    #print(arr)
    arr = sorted(arr, key = lambda x : x[k])
    #print(arr)
    for i in range(n):
        for j in range(m):
            print(arr[i][j], end = " ")
        print("")

#------------------------- ginortS ---------------------------------------------

s = input()
lc = []
uc = []
od = []
ed = []
for i in range(len(s)):
    if s[i].isdigit():
        if int(s[i])% 2 == 0:
            ed.append(s[i])
        else:
            od.append(s[i])
    elif s[i].islower():
        lc.append(s[i])
    elif s[i].isupper():
        uc.append(s[i])
out = ""
lc.sort()
uc.sort()
od.sort()
ed.sort()
for item in lc:
    out += item
for item in uc:
    out += item
for item in od:
    out += item
for item in ed:
    out += item
print(out)
        
#------------------------- Calendar Module -------------------------------------

import calendar

m, d, y = map(int, input().split())
i = calendar.weekday(y, m, d)
l = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
print(l[i])

#------------------------- Time Delta ------------------------------------------

#!/bin/python3

import math
import os
import random
import re
import sys
from datetime import datetime, timezone
# Complete the time_delta function below.
def time_delta(t1, t2):
    tmp = t1.split()
    t1_dt = datetime.strptime(' '.join(tmp[1:]), '%d %b %Y %H:%M:%S %z')
    t1_dt_utc = t1_dt.astimezone(tz = timezone.utc)
    tmp = t2.split()
    t2_dt = datetime.strptime(' '.join(tmp[1:]), '%d %b %Y %H:%M:%S %z')
    t2_dt_utc = t2_dt.astimezone(tz = timezone.utc)
    return str(abs((t1_dt_utc - t2_dt_utc).days * 24 * 3600 + (t1_dt_utc - t2_dt_utc).seconds))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()

#------------------------- collections.Counter() -------------------------------

from collections import Counter
x = int(input())
shoes = list(map(int, input().split()))
c = Counter(shoes)
earnings = 0
for i in range(int(input())):
    size, price = map(int, input().split())
    if c[size] > 0:
        c[size] -= 1
        earnings += price
print(earnings)

#------------------------- DefaultDict Tutorial --------------------------------

from collections import defaultdict
n, m = map(int, input().split())
a = defaultdict(list)
b = []
for i in range(n):
    a[input()].append(i+1)
for i in range(m):
    item = input()
    if a[item] == []:
        a[item].append(-1)
    print(" ".join(map(str,a[item])))

#------------------------- Collections.namedtuple() ----------------------------

from collections import namedtuple

n = int(input())
Student = namedtuple('Student', input())
students = []
for i in range(n):
    args = input().split()
    st = Student(args[0], args[1], args[2], args[3])
    students.append(st)
s = 0
for item in students:
    s += int(item.MARKS)
print(s/n)
    
#------------------------- Collections.OrderedDict() ---------------------------

from collections import OrderedDict
n = int(input())
od = OrderedDict()
for i in range(n):
    row = input().split()
    price = int(row[-1])
    name = ' '.join(row[:-1])
    if od.__contains__(name):
        od[name] += price
    else:
        od[name] = price
for item in od:
    print(item, od[item])

#------------------------- Word Order ------------------------------------------

from collections import OrderedDict
n = int(input())
od = OrderedDict()
for i in range(n):
    word = input()
    if od.__contains__(word):
        od[word] += 1
    else:
        od[word] = 1
print(len(od))
for item in od:
    print(od[item], end=" ")
    
#------------------------- Collections.deque() ---------------------------------

from collections import deque

n = int(input())
d = deque()
for i in range(n):
    command = input().split()
    if command[0] == "append":
        d.append(int(command[1]))  
    elif command[0] == "appendleft":  
        d.appendleft(int(command[1]))  
    elif command[0] == "pop":
        d.pop()  
    elif command[0] == "popleft":
        d.popleft()
for i in d:
    print(i, end = " ")

#------------------------- Company Logo ----------------------------------------

from collections import Counter

if __name__ == '__main__':
    s = input()
    c = Counter(s.replace(" ", ""))
    n = 3
    while n > 0:
        l = []
        m = max(c.values())
        for item in list(c):
            if c[item] == m:
                x = c.pop(item)
                l.append(item)
        l.sort()
        for i in l:
            if n > 0:
                print(str(i) + " " + str(m))
                n -= 1

#------------------------- Piling Up! ------------------------------------------

from collections import deque
t = int(input())
for i in range(t):
    d = deque()
    flag = True
    n = int(input())
    l = list(map(int, input().split()))
    for item in l:
        d.append(item)
    temp = 0
    if d[0] >= d[-1]:
        temp = d.popleft()
    else:
        temp = d.pop() 
    for k in range(n-2):
        if d[0] >= d[-1]:
            temp2 = d.popleft()
        else:
            temp2 = d.pop()
        if(temp < temp2):
            flag = False
        temp = temp2
    if d[0] <= temp and flag == True:
        print("Yes")
    else:
        print("No")

#------------------------- Introduction to Sets --------------------------------

def average(array):
    # your code goes here
    s = set(arr)
    return sum(s)/len(s) # It's not even correct but it's what the question asks for, so...

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)
        
#------------------------- No Idea! --------------------------------------------

n, m = map(int, input().split())
arr = list(map(int, input().split()))
a = set(map(int, input().split()))
b = set(map(int, input().split()))
happiness = 0
for item in arr:
    if a.__contains__(item):
        happiness += 1
    elif b.__contains__(item):
        happiness -= 1
print(happiness)

#------------------------- Symmetric Difference --------------------------------

m = int(input())
set1 = set(map(int, input().split()))
n = int(input())
set2 = set(map(int, input().split()))
#sd = (set1.union(set2)).difference(set1.intersection(set2))
sd = set1.__xor__(set2)
for i in range(len(sd)):
    print(min(sd))
    sd.remove(min(sd))

#------------------------- Set .add() ------------------------------------------

n = int(input())
countries = set()
for i in range(n):
    countries.add(input())
print(len(countries))
    
#------------------------- Set .discard(), .remove() & .pop() ------------------

n = int(input())
s = set(map(int, input().split()))
nc = int(input())
for i in range(nc):
    command = input().split()
    if command[0] == 'pop':
        s.pop()
    elif command[0] == 'remove':
        s.remove(int(command[1]))
    elif command[0] == 'discard':
        s.discard(int(command[1]))
print(sum(s))

#------------------------- Set .union() Operation ------------------------------

input()
n = set(map(int, input().split()))
input()
b = set(map(int, input().split()))
print(len(n.union(b)))

#------------------------- Set .intersection() Operation -----------------------

input()
n = set(map(int, input().split()))
input()
b = set(map(int, input().split()))
print(len(n.intersection(b)))

#------------------------- Set .difference() Operation -------------------------

input()
e = set(map(int, input().split()))
input()
f = set(map(int, input().split()))
print(len(e-f))

#------------------------- Set .symmetric_difference() Operation ---------------

input()
e = set(map(int, input().split()))
input()
f = set(map(int, input().split()))
print(len(e^f))

#------------------------- Set Mutations ---------------------------------------

input()
a = set(map(int, input().split()))
for i in range(int(input())):
    command, n = input().split()
    s = set(map(int, input().split()))
    if command == "update":
        a.update(s)
    elif command == "intersection_update":
        a.intersection_update(s)
    elif command == "difference_update":
        a.difference_update(s)
    elif command == "symmetric_difference_update":
        a.symmetric_difference_update(s)
print(sum(a))

#------------------------- The Captain's Room ----------------------------------

k = int(input())
rnl = list(map(int, input().split()))
s = set(rnl)
print( int((sum(s)*k-sum(rnl))/(k-1)) )

#------------------------- Check Subset ----------------------------------------

for i in range(int(input())):
    input()
    a = set(map(int, input().split()))
    input()
    b = set(map(int, input().split()))
    if a.issubset(b):
        print("True")  
    else:
        print("False") 

#------------------------- Check Strict Superset -------------------------------

a = set(map(int, input().split()))
flag = True
for i in range(int(input())):
    s = set(map(int, input().split()))
    if not(a.issuperset(s)) or len(a) <= len(s):
        flag = False
print(flag)
        
#------------------------- sWAP cASE -------------------------------------------

def swap_case(s):
    swapedstr = ""
    for i in range(len(s)):
        if s[i].islower():
            swapedstr += s[i].upper()
        else:
            swapedstr += s[i].lower()
    return swapedstr

if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)

#------------------------- String Split and Join -------------------------------

def split_and_join(line):
    # write your code here
    lst = line.split(" ")
    return "-".join(lst)

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

#------------------------- What's Your Name? -----------------------------------

def print_full_name(a, b):
    print("Hello " + a + " " + b + "! You just delved into python.")

if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)

#------------------------- Mutations -------------------------------------------

def mutate_string(string, position, character):
    return string[:position] + character + string[position+1:]

if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)
    
#------------------------- Find a string ---------------------------------------

def count_substring(string, sub_string):
    cnt = 0
    for i in range(len(string)-len(sub_string)+1):
        if string[i:i+len(sub_string)] == sub_string:
            cnt += 1
    return cnt

if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()
    
    count = count_substring(string, sub_string)
    print(count)

#------------------------- String Validators -----------------------------------

if __name__ == '__main__':
    s = input()
    alphanum = False
    alpha = False
    dig = False
    lc = False
    uc = False
    for i in range(len(s)):
        if s[i].isalnum():
            alphanum = True
        if s[i].isalpha():
            alpha = True
        if s[i].isdigit():
            dig = True
        if s[i].islower():
            lc = True
        if s[i].isupper():
            uc = True
    print(alphanum)
    print(alpha)
    print(dig)
    print(lc)
    print(uc)
    
#------------------------- Text Alignment --------------------------------------

#Replace all ______ with rjust, ljust or center. 

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))
    
#------------------------- Text Wrap -------------------------------------------

import textwrap

def wrap(string, max_width):
    return textwrap.fill(string, max_width)

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)
    
#------------------------- Designer Door Mat -----------------------------------

n, m = input().split()
n = int(n)
m = int(m)
c = ".|."
for i in range(n//2):
    print( (c*i).rjust(int((m-3)/2), '-')+c+(c*i).ljust(int((m-3)/2), '-') )
print("WELCOME".center(m, '-'))
for i in range(n//2-1, -1, -1):
    print( (c*i).rjust(int((m-3)/2), '-')+c+(c*i).ljust(int((m-3)/2), '-') )

#------------------------- String Formatting -----------------------------------

def print_formatted(number):
    # your code goes here
    w = len(bin(number))-2
    for i in range(1, number +1):
       print(("{0:"+ str(w) +"d} {1:"+ str(w) +"o} {2:"+ str(w) +"X} {3:"+ str(w) +"b}").format(i, i, i, i))
       
if __name__ == '__main__':
    n = int(input())
    print_formatted(n)
    
#------------------------- Alphabet Rangoli ------------------------------------

import string
def print_rangoli(size):
    for i in range(size -1, -1, -1):
        s = string.ascii_lowercase[i]
        for j in range(i+1, size):
            s = string.ascii_lowercase[j] + "-" + s + "-" + string.ascii_lowercase[j]
        print(s.center(4*size -3, '-'))
    for i in range(1, size):
        s = string.ascii_lowercase[i]
        for j in range(i+1, size):
            s = string.ascii_lowercase[j] + "-" + s + "-" + string.ascii_lowercase[j]
        print(s.center(4*size -3, '-'))
        
if __name__ == '__main__':
    n = int(input())
    print_rangoli(n)
    
#------------------------- Capitalize! -----------------------------------------

# Complete the solve function below.
def solve(s):
    # l = s.split()
    # for i in range(len(l)):
    #     l[i] = l[i][0].upper() + l[i][1:]
    # return " ".join(l)
    flag = True
    for i in range(len(s)):
        if s[i] == ' ':
            flag = True
        elif flag == True:
            s = s[:i] + s[i].upper() + s[i+1:]
            flag = False
    return s

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = solve(s)

    fptr.write(result + '\n')

    fptr.close()

#------------------------- The Minion Game -------------------------------------

def minion_game(string):
    # your code goes here
    lst = ['A', 'E', 'I', 'O', 'U']
    stuart = 0
    kevin = 0
    for i in range(len(string)):
        if lst.__contains__(s[i]):
            kevin += len(string) -i
        else:
            stuart += len(string) -i
    if stuart > kevin:
        print("Stuart " + str(stuart))
    elif stuart < kevin:
        print("Kevin " + str(kevin))
    else:
        print("Draw")
        
if __name__ == '__main__':
    s = input()
    minion_game(s)
    
#------------------------- Merge the Tools! ------------------------------------

def merge_the_tools(string, k):
    # your code goes here
    n = len(string)
    s = string[0]
    for i in range(1, n):
        if i % k == 0:
            print(s)
            s = string[i]
        else:
            flag = False
            for j in range(len(s)):
                if s[j] == string[i]:
                    flag = True
            if flag == False:
                s += string[i]
    print(s)
    
if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)

#------------------------- List Comprehensions ---------------------------------

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    print([ [ i, j, k] for i in range(x + 1) for j in range(y + 1) for k in range(z + 1) if ( ( i + j + k ) != n )])
    
#------------------------- Find the Runner-Up Score! ---------------------------

if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    ls = list(arr)
    ls.sort()
    mx = ls[-1]
    while ls[-1] == mx:
        ls = ls[:-1]
    print(ls[-1])
    
#------------------------- Nested Lists ----------------------------------------

if __name__ == '__main__':
    d = {}
    for _ in range(int(input())):
        name = input()
        score = float(input())
        d[name] = score
    minimum = min(d.values())
    delete = [key for key in d if d[key] == minimum] 
    for key in delete:
        del d[key] 
    minimum = min(d.values())
    lst = []
    for key, value in d.items():
        if value == minimum:
            lst += [key]
    lst.sort()
    for i in lst:
        print(i)
        
#------------------------- Finding the percentage ------------------------------

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    ave = sum(student_marks[query_name])/3
    print("{:.2f}".format(ave))

#------------------------- Lists -----------------------------------------------

if __name__ == '__main__':
    N = int(input())
    commands = []
    for _ in range(N):
        func, *line = input().split()
        args = list(map(int, line))
        commands += [[func, args]]
    lst = []
    for i in commands:
        if i[0] == 'insert':
            lst.insert(i[1][0], i[1][1])
        elif i[0] == 'print':
            print(lst)
        elif i[0] == 'remove':
            lst.remove(i[1][0])
        elif i[0] == 'append':
            lst.append(i[1][0])
        elif i[0] == 'sort':
            lst.sort()
        elif i[0] == 'pop':
            lst.pop()
        elif i[0] == 'reverse':
            lst.reverse()
            
#------------------------- Tuples ----------------------------------------------

if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    t = tuple(integer_list)
    print(t.__hash__())
    
#------------------------- Say "Hello, World!" With Python ---------------------

print("Hello, World!") #Do I really need to do this? :D
    
#------------------------- Python If-Else --------------------------------------

#!/bin/python3

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input().strip())
    if n % 2 == 1 or ( n > 5 and n < 21 ):
        print("Weird")
    else:
        print("Not Weird")
        
#------------------------- Arithmetic Operators --------------------------------

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a+b)
    print(a-b)
    print(a*b)
    
#------------------------- Python: Division ------------------------------------

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a//b)
    print(a/b)
    
#------------------------- Loops -----------------------------------------------

if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        print(i**2)

#------------------------- Write a function ------------------------------------

def is_leap(year):
    leap = False
    
    # Write your logic here
    if year % 4 == 0:
        if year % 400 == 0:
            leap = True
        elif (year % 100) == 0:
            leap = False
        else:
            leap = True

    return leap

year = int(input())
print(is_leap(year))
        
#------------------------- Print Function --------------------------------------

if __name__ == '__main__':
    n = int(input())
    str1 = ""
    for i in range(1, n+1):
        str1 += str(i)
    print(str1)
