# Step 1: ask for numbers, alert order matters for sunstractign and dividing
print("please keep in mind that the order of your numbers matter when perfoming subtractions and divisions")
x = float(input('Enter the first number with decimal: '))
y = float(input('Enter the second number with decimal : '))

# step 2: ask user for operation to be done 
operation = input("would you like to add/subtract/multiply/divide?: ").lower()
# print("you chose {}.".format(operation)) #testing purposes


if operation == "divide":
    ans = x / y 
    print("your quotient is:", ans)
elif operation == 'add':
    ans = x + y
    print("Your sum is: ", ans)
elif operation == 'subtract':
    diff = x - y   
    print(" The difference is: ", diff) 
elif operation == 'multiply':
    product = x * y
    print("The product is: ", product)
else:
    print("Sorry {} operation is not yet supported.".format(operation))

