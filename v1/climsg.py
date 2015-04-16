from colorama import init, Fore, Back
from termcolor import colored

init()

def welcome_message(script_name):
    print (Fore.CYAN + "--------------------------------------------------------")
    print "---------     EEG classification challenge    ----------"
    print "--------------------------------------------------------"
    print ""
    print "                        ~                               "
    print (Fore.RESET +"")
    print("Executing script : [>" + colored(script_name,'green') + "]")
    print ""
    print ""

def loading_data():
    print (colored('[Loading data]','red'))

def done_loading_data(duration):
    print('  > Done loading data in ' + colored(str(duration),'green') + ' seconds.')

def features_building_init():
    print (colored('[Features Construction]','blue'))

def freq_features(duration,nb_features):
    print ("  > Frequency features building done in : " + colored(str(duration),'green') + ' seconds, for ' + colored(str(nb_features),'blue') + ' features')

def wav_features(duration,nb_features):
    print ("  > Wavelets features building done in : " + colored(str(duration),'green') + ' seconds, for ' + colored(str(nb_features),'blue') + ' features')

def stat_features(duration,nb_features):
    print ("  > Static features building done in : " + colored(str(duration),'green') + ' seconds, for ' + colored(str(nb_features),'blue') + ' features')

def predict():
    print (colored('[Predicting]','red'))

def done_predicting(duration):
    print ("  > Done predicting on the test set in " + colored(str(duration),'green') + " seconds")

def report():
    print ""
    print (Fore.RED + "--------------------------------------------------------")
    print "-------------       Performance           --------------"
    print "--------------------------------------------------------"
    print (Fore.RESET +"")

def export():
    print ""
    print (Fore.MAGENTA + "--------------------------------------------------------")
    print "-------------          Submit             --------------"
    print "--------------------------------------------------------"
    print (Fore.RESET +"")
    print(colored('[Export]','red'))

def done_export(duration):
    print ("  > Done exporting in " + colored(str(duration),'green') + " seconds")

def goodbye(duration):
    print ""
    print (Fore.YELLOW + "--------------------------------------------------------")
    print "-------------       Completed!           ---------------"
    print "--------------------------------------------------------"
    print (Fore.RESET +"")
    print("Total time execution : " + colored(str(duration),'yellow') + " seconds.")
    print "Goodbye."
