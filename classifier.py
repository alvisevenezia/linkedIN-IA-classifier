from transformers import pipeline,AutoModelForSequenceClassification, TrainingArguments, Trainer
import csv
import random
import signal
import msvcrt

label_id = 8
text_id = 3

global classifier

labels = ["Not IA","Surely NOT IA","Maybe IA","Surely IA","IA"]

#matrix to store the results
results_matrix = [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]

id2label = {0: "Not IA", 1: "Surely NOT IA", 2: "Maybe IA", 3: "Surely IA", 4: "IA"}
label2id = {"Not IA": 0, "Surely NOT IA": 1, "Maybe IA": 2, "Surely IA": 3, "IA": 4}

def roberta_classify(text, candidate_labels):
    
    result = classifier(text, candidate_labels)
    return result
 
def save_results():
    #write the results to a file
    with open("results.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["sep=,"])
        writer.writerow(["Prédictions : Réels"]+labels)
        for i in range(0, len(labels)):
            writer.writerow([labels[i]]+results_matrix[i])

        #put statistics in the file
        writer.writerow([" "])
        writer.writerow([" "])

        writer.writerow(["Statistics"])
        writer.writerow([" "])
        writer.writerow([" "]+labels)

        for i in range(0, len(labels)):
            total = sum(results_matrix[i]) 
            row = [" "]*(len(labels)+2)
            row[0] = "Total row " + str(i) +" : "+str(total)
            for j in range(0, len(labels)):
                #print porcentage 
                row[j+1] = str(round(results_matrix[i][j]*100/(total if total != 0 else 1),2)) + " %"

            row[len(labels)+1] = labels[i]
            writer.writerow(row)


    print("Saved")

def exit_handler(signal, frame):
    print("Exiting...")
    #write the results to a file
    save_results()
    #exit the program
    exit(0)

def save_handler(signal, frame):
    print("Saving...")
    #write the results to a file
    save_results()

    print("Saved")

def main():
    
    global classifier

    signal.signal(signal.SIGINT, exit_handler)

    #ask user for method and  nbmber of rows to classify
    print("Starting...")
    print(" ")
    print("Select the method to classify the text: ")
    print(" ")
    print("1. Roberta")
    print("2. CamemBERT")
    print(" ")

    method = input("Method: ")

    print(" ")

    try :
        max_rows = int(input("Number of rows to classify: "))
    except ValueError:
        print("Invalid input")
        return

    if method == "1":
        classifier = pipeline('text-classification',model='roberta-large-mnli')

    elif method == "2":

        model = AutoModelForSequenceClassification.from_pretrained(
            "camembert-base",num_labels=len(labels),id2label=id2label, label2id=label2id
        )

        classifier = pipeline('text-classification',model=model)


    #open file.csv with xlsxwriter
    file = open("file.csv", "r", encoding="utf-8")
    
    #read the file
    reader = csv.reader(file)

    n = 0

    #skip the header
    next(reader)

    #start at a random row
    random_start = random.randint(0, 11000-max_rows)

    print("Random start: ", random_start)

    for i in range(0, random_start):
        next(reader)

    print("Starting...")

    for row in reader:

        if msvcrt.kbhit():
            #if the user presses wtrl + s, save the results
            if msvcrt.getch() == b'\x13':
                save_handler(0, 0)

        #check if the row is empty
        if row == []:
            continue

        text = row[text_id]
        true_label = row[label_id]

        if true_label == "":
            continue

        #classify the text

        result = roberta_classify(text, labels)
    
        print(result)

        predicted_label = result['labels'][0]
        predicted_label_index = labels.index(predicted_label)

        print("n: ", n)
        print("Text: ", text)
        print("True label: ", true_label)
        print("Predicted label: ", predicted_label)
        print("Predicted label index: ", predicted_label_index)
        print("Result: ", result)
        print(" ")

        #append the results to the matrix
        results_matrix[int(float(true_label))][predicted_label_index] += 1
                  
        n += 1

        if n == max_rows:
            break

        print(" ")

    save_results()

    #convert the matrix to porcentage
    for i in range(0, len(labels)):
        total = sum(results_matrix[i]) 
        print ("Total row " + str(i) +" : "+str(total)) 
        for j in range(0, len(labels)):

            results_matrix[i][j] = results_matrix[i][j]*100/(total if total != 0 else 1)

    for i in range(0, len(labels)): 
        print(results_matrix[i])

if __name__ == "__main__":
    main()