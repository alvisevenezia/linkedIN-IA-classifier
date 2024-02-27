import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Création d'un jeu de données pour l'exemple
data_example = {
    "experience_description": [
        "Developed a new algorithm for facial recognition using deep learning",
        "Managed a team of software developers in a tech company",
        "Designed and implemented a database system for a hospital",
        "Conducted research on machine learning applications in finance",
        "Worked on a project using natural language processing to analyze texts",
        "Taught mathematics and statistics at a university",
        "Developed a mobile application for a startup",
        "Worked in a lab on robotics and automation systems",
        "Participated in a cybersecurity workshop",
        "Conducted market analysis for a new product launch"
    ],
    "label": [1, 0, 0, 1, 1, 0, 0, 1, 0, 1]  # 1 pour IA, 0 sinon
}

# Conversion en DataFrame et sauvegarde en fichier CSV
df_example = pd.DataFrame(data_example)
csv_file_example_path = 'C:/Users/emman/Desktop/example_ai_experience.csv'
df_example.to_csv(csv_file_example_path, index=False)

# Fonction pour la classification binaire des expériences professionnelles
def ai_experience_classifier(csv_path):
    # Charger les données
    data = pd.read_csv(csv_path)

    # Préparation des données
    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(data['experience_description'])
    y = data['label']

    # Division en ensembles d'entraînement et de test avec une stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Création et entraînement du modèle Naive Bayes
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Prédiction et évaluation
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Affichage de la matrice de confusion
    conf_matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de confusion')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # Affichage d'un rapport de classification
    print(classification_report(y_test, predictions, target_names=['Non-IA', 'IA']))

    return accuracy

# Appel de la fonction avec le chemin du fichier CSV
accuracy = ai_experience_classifier(csv_file_example_path)
print(f"The accuracy of the model is: {accuracy}")

