import sys as sy, os as os

def clear():
    if sy.platform == 'win32':
        os.system('cls')
    else:
        os.system('clear')

def sort(x):
    # x = sy.argv[1] if len(sy.argv) > 1 else None

    clear()

    if x is None:
        return "Error: No input provided."
    else:
        try:
            changed = 1 == 1
            x = list(x)
            num = 0
            while changed:
                if num == len(list(x)):
                    num = 0

                if x[num - 1] > x[num]:
                    changed = True
                    temp = x[num]
                    x[num] = x[num - 1]
                    x[num - 1] = temp
                else:
                    changed = False
                    
            return x
        except Exception as e:
            clear()
            return f"Error: {e}"