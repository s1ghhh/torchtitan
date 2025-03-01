
def generate_list(n):
    result = []
    for i in range(n):
        if i % 2 == 0:
            result.append("*#")
        else:
            result.append("#")
    return result

n = 28
generated_list = generate_list(n)
print(generated_list)
