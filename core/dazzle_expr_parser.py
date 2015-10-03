"""
A pyparsing definition of  Numexpr parser: https://github.com/pydata/numexpr

This parser is not specially fast (almost 0.1s for parsing the 1 line expr at the end of this file.
Nor it is precise for error detection.
Might be useful to re-implement that using ANTLR-python

"""

from __future__ import division
from pyparsing import *
import pyparsing

if not ParserElement._packratEnabled:
    ParserElement._packratEnabled = True
    ParserElement._parse = ParserElement._parseCache

class StringLiteral:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return "<" + self.value + ">"

class Ident:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "Ident(" + self.name + ")"

def convert_number(s):
    if (s[0] in ('-', '+') and s[1:].isdigit()) or s.isdigit():
        return int(s)
    else:
        return float(s)

def make_ident(s):
    return Ident(s)

def make_string_literal(name):
    return StringLiteral(name)

expr = Forward()

integer = Word(nums).setParseAction(lambda t:int(t[0]))
point = Literal( "." )
string = QuotedString("'", escChar='\\').setParseAction(lambda t: make_string_literal(t[0]))
e     = CaselessLiteral( "E" )
number = Combine( Word( "+-"+nums, nums ) +
                   Optional( point + Optional( Word( nums ) ) ) +
                   Optional( e + Word( "+-"+nums, nums ) ) ).setParseAction(lambda t: convert_number(t[0]))
ident = Word(alphas, alphas+nums+"_$").setParseAction(lambda t: make_ident(t[0]))
lpar  = Literal( "(" ).suppress()
rpar  = Literal( ")" ).suppress()
comma  = Literal( "," ).suppress()
funident1 = oneOf("abs sqrt exp expm1 sin cos tan arcsin arccos arctan arctan2 sinh cosh tanh arcsinh arccosh arctanh log log10 log1p upper lower")
funident2 = oneOf("concat")
funident3 = oneOf("where substring")
funcall1 = funident1 + lpar + expr + rpar
funcall2 = (funident2 + lpar + expr + comma +  expr + rpar)
funcall3 = (funident3 + lpar + expr + comma + expr + comma +  expr + rpar)
operand = number | string | funcall1 | funcall2 | funcall3 | ident

signop = oneOf('+ -')
multop = oneOf('* / %')
addop = oneOf('+ -')
shiftop = oneOf('<< >>')
boolop = oneOf('& |')
notop = Literal('~')
expop = Literal('**')
compop = oneOf('< <= > >=  == !=')

expr << operatorPrecedence( operand,
    [
     (notop, 1, opAssoc.RIGHT),
     (signop, 1, opAssoc.RIGHT),
     (expop, 2, opAssoc.RIGHT),
     (multop, 2, opAssoc.LEFT),
     (addop, 2, opAssoc.LEFT),
     (shiftop, 2, opAssoc.LEFT),
     (boolop, 2, opAssoc.LEFT),
     (compop, 2, opAssoc.LEFT),
    ]
    )

if __name__ == '__main__':
    import time


    def test1():
        #test = "3 * (2.0 + where(xhjuygty >= 1 & ~ (sin(whoooooo) == -58434535 | y > 3), 1, 2)) ** 2"
        test = "concat(concat(upper(substring(firstname, 0, (2 % 3) + 1)), '. '), lastname)"
        t1 = time.time()

        print(expr.parseString(test, parseAll=True))

        print(time.time() - t1)

    test1()
