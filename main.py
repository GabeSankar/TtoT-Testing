# This is a sample Python script.
import Bert as bt
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



def run(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    model = bt.Bert(vocabFile="cased30k.vocab", wordPiece=True, onlyLowercase=False)
    #model.train(500)
    #model.metricTest()
    model.printExample(model.test_dataset[142])
    #print(model.test(model.test_dataset[1]))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    run('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
