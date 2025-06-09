# a,b,*rest = range(5)
# print(rest)

# def fun(a, b, c, d, *rest):
#     return a, b, c, d, rest

# print(fun(*[1,2],3,*range(4,7)))

# print(*range(4),4)  


# metro_areas = [
# ('Tokyo', 'JP', 36.933, (35.689722, 139.691667)), 
# ('Delhi NCR', 'IN', 21.935, (28.613889, 77.208889)),
# ('Mexico City', 'MX', 20.142, (19.433333, -99.133333)),
# ('New York-Newark', 'US', 20.104, (40.808611, -74.020386)),
# ('São Paulo', 'BR', 19.649, (-23.547778, -46.635833))
# ]
# def jiebao(metro_areas):
#     print(f"{"":15}|{"latitude":>9}|{"longitude":>9}")
#     for name,_,_,(lat,lon) in metro_areas:
#         if lon < 0:
#             print(f"{name:15}|{lat:9.4f}|{lon:9.4f}")

# jiebao(metro_areas)


# match case coding
def handle_command(self,message):
    match message:
        case ['BEEPER', frequency, times]: 
            self.beep(times, frequency)
        case ['NECK',angle]:
            self.rotate_neck(angle)
        case _:
            raise InvaildCommand(message)

metro_areas = [
('Tokyo', 'JP', 36.933, (35.689722, 139.691667)),
('Delhi NCR', 'IN', 21.935, (28.613889, 77.208889)),
('Mexico City', 'MX', 20.142, (19.433333, -99.133333)),
('New York-Newark', 'US', 20.104, (40.808611, -74.020386)),
('São Paulo', 'BR', 19.649, (-23.547778, -46.635833)),
]
def case_study():
    print(f'{"":15} | {"latitude":>9} | {"longitude":>9}')
    for record in metro_areas:
        match record: 
            case [name, _, _, (lat, lon)] if lon <= 0: 
                print(f'{name:15} | {lat:9.4f} | {lon:9.4f}')

case_study()

 # use python to get a explanation of the code
def evaluation(exp,env)-> any:
    "evalution the expression in the environment"
    match exp:
        case ['quote', x]:
            return x
        case ['if', test, consequence, alternative]:
            if evaluation(test, env):
                return evaluation(consequence, env)
            else:
                return evaluation(alternative, env)
        case ['lambda', [*params], *body] if body:
            return Procedure(params, body, env)
        case ['define',Symbol as name,value_exp]:
            env[name] = evaluation(value_exp, env)
            return None
        case _:
            raise SyntaxError(exp)

print(evaluation(["quote", 5],{"env":10})) 
# print(evaluation(['if',10,15,20],{"env":10})) # Example usage of the evaluation function

# slice 
# 切片不去最后一项，可以容易判断切片的长度
# 切片文字，最容易遇到
s = "bycycle"
print(s[::-1])
print(s[::-2])   

l =list(range(10))
print(l)
l[2:5] = [20,30]
print(l)
del l[5:7]
print(l)
print(l[3::2])
print(l)
