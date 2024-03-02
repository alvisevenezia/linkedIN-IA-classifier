from transformers import pipeline,AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer,DataCollatorWithPadding
import csv
import random
import signal
import msvcrt
from datasets import load_dataset
import numpy as np
import evaluate

label_id = 8
text_id = 3
train_data_index = random.randint(0, 4000)
test_data_index = random.randint(4500 , 11000)


accuracy = evaluate.load("accuracy")

global classifier
global label

tokeniser = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

labels_en = ["Not IA","Surely NOT IA","Maybe IA","Surely IA","IA"]
label_fr = ["Pas IA","Sûrement PAS IA","Peut-être IA","Sûrement IA","IA"]

#matrix to store the results
results_matrix = [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]

id2label_en = {0: "Not IA", 1: "Surely NOT IA", 2: "Maybe IA", 3: "Surely IA", 4: "IA"}
label2id_en = {"Not IA": 0, "Surely NOT IA": 1, "Maybe IA": 2, "Surely IA": 3, "IA": 4}

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def clean_data(data):

    print("Cleaning data...")
    
    row_erased = 0 

    total = len(data["train"])

    #iterate through the data and remove the rows with empty values
    for i in range(0, len(data["train"])):

        printProgressBar(i, total, prefix = 'Training dataset cleaned : |', suffix = 'Complete', length = 50)

        #check if there is a NaN value in the row
        if data["train"][i]["jobTitle"] == None:
           
            #remove the row
            data["train"][i]["jobTitle"] = "test"

            row_erased += 1

    print("Data cleaned :" + str(row_erased) + " rows erased")

    return data


def preprocess_function(data):

    return tokeniser(text=data["jobTitle"], padding=True, truncation=True, max_length=128)

def train_model():

    return

def roberta_classify(text, candidate_labels):
    
    result = classifier(text, candidate_labels)
    return result
 
def save_results():
    #write the results to a file
    with open("results.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["sep=,"])
        writer.writerow(["Prédictions : Réels"]+label)
        for i in range(0, len(label)):
            writer.writerow([label[i]]+results_matrix[i])

        #put statistics in the file
        writer.writerow([" "])
        writer.writerow([" "])

        writer.writerow(["Statistics"])
        writer.writerow([" "])
        writer.writerow([" "]+label)

        for i in range(0, len(label)):
            total = sum(results_matrix[i]) 
            row = [" "]*(len(label)+2)
            row[0] = "Total row " + str(i) +" : "+str(total)
            for j in range(0, len(label)):
                #print porcentage 
                row[j+1] = str(round(results_matrix[i][j]*100/(total if total != 0 else 1),2)) + " %"

            row[len(label)+1] = label[i]
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

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def main():
    
    global classifier
    global label
    
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
        label = labels_en

        dataset = load_dataset("csv", data_files="file.csv")

        #ony keep collumn "jobTitle" and "label"
        dataset = dataset.remove_columns(["id","companyName","description","companyUrl","dateRange","location","logoUrl"])

        dataset = dataset["train"].train_test_split(test_size=0.2)
        tokenised_dataset = dataset.map(preprocess_function,batched=True)

        data_collator = DataCollatorWithPadding(tokenizer=tokeniser)

        model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-large-mnli", num_labels=5, id2label=id2label_en, label2id=label2id_en, ignore_mismatched_sizes=True
            )
        
        training_args = TrainingArguments(
            output_dir="./trained_models/roberta",
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=2,
            num_train_epochs=2,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            label_names=labels_en,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenised_dataset["train"],
            eval_dataset=tokenised_dataset["test"],
            tokenizer=tokeniser,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        trainer.train()

        classifier = pipeline('text-classification',model='roberta-large-mnli')

    elif method == "2":

        #model = AutoModelForSequenceClassification.from_pretrained("camembert-base", num_labels=5, id2label=id2label, label2id=label2id)

        label = label_fr
        classifier = pipeline('zero-shot-classification',model='camembert-base')


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

        result = roberta_classify(text, label)
    
        print(result)

        predicted_label = result['labels'][0]
        predicted_label_index = label.index(predicted_label)

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
    for i in range(0, len(label)):
        total = sum(results_matrix[i]) 
        print ("Total row " + str(i) +" : "+str(total)) 
        for j in range(0, len(label)):

            results_matrix[i][j] = results_matrix[i][j]*100/(total if total != 0 else 1)

    for i in range(0, len(label)): 
        print(results_matrix[i])

if __name__ == "__main__":
    main()