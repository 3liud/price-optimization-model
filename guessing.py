# guessing game
from random import randint
# from IPython.display import clear_output

player = input("Hello, enter you name: ")
guessed = False
number = randint(0, 10)
guesses = 0
while not guessed:
    ans = input(f"hello {player}! try to guess the NUmber I am thinking of: ")
#    ans = input("Hello {}! try to guess the Number I am thinking of! ".format(player))
    guesses += 1
#    clear_output()
    if int(ans) == number:
        print("congrats! {} You guessed it correctly.".format(player))
        print("It took you {} guesses!".format(guesses))
        break
    elif int(ans) > number:
        print("The number is lower than what you guessed.")
    elif int(ans) < number:
        print("The number is greater than what you guessed.")
