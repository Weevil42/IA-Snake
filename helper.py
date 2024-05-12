import matplotlib.pyplot as plt  # Importe la bibliothèque Matplotlib pour créer des graphiques
from IPython import display  # Importe la fonction display du module IPython pour manipuler les sorties d'affichage dans un notebook IPython

plt.ion()  # Active le mode interactif de Matplotlib pour permettre les mises à jour en temps réel des graphiques

def plot(scores, mean_scores):
    display.clear_output(wait=True)  # Efface la sortie d'affichage actuelle dans le notebook IPython
    display.display(plt.gcf())  # Affiche la figure actuelle de Matplotlib
    plt.clf()  # Efface le contenu de la figure actuelle de Matplotlib
    plt.title('Entrainement en cours...')  # Définit le titre du graphique
    plt.xlabel('Nombre de parties')  # Définit le titre de l'axe des abscisses
    plt.ylabel('Score')  # Définit le titre de l'axe des ordonnées
    plt.plot(scores)  # Trace la courbe des scores bruts
    plt.plot(mean_scores)  # Trace la courbe des moyennes des scores
    plt.ylim(ymin=0)  # Définit la limite inférieure de l'axe des ordonnées à 0
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))  # Ajoute un texte pour afficher le dernier score brut
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))  # Ajoute un texte pour afficher la dernière moyenne de score
    plt.show(block=False)  # Affiche le graphique en cours de construction
    plt.pause(.1)  # Pause l'exécution du code pendant 0.1 seconde pour permettre à Matplotlib de mettre à jour l'affichage
