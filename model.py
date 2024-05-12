import torch  
# Importation de la bibliothèque PyTorch pour le calcul tensoriel
import torch.nn as nn  
# Importation des modules de réseaux de neurones de PyTorch
import torch.optim as optim  
# Importation du module d'optimisation de PyTorch
import torch.nn.functional as F  
# Importation des fonctions d'activation et de coût de PyTorch
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()  # Initialisation de la classe parente
        self.linear1 = nn.Linear(input_size, hidden_size)  # Définition de la couche linéaire 1
        self.linear2 = nn.Linear(hidden_size, output_size)  # Définition de la couche linéaire 2

    def forward(self, x):
        x = F.relu(self.linear1(x))  # Application de la fonction d'activation ReLU à la sortie de la couche linéaire 1
        x = self.linear2(x)  # Passage à travers la couche linéaire 2
        return x  # Renvoi de la sortie

    def save(self, file_name='model.pth'):
        model_folder_path = './model'  # Définition du chemin du dossier de sauvegarde du modèle
        if not os.path.exists(model_folder_path):  # Vérification si le dossier existe
            os.makedirs(model_folder_path)  # Création du dossier s'il n'existe pas

        file_name = os.path.join(model_folder_path, file_name)  # Définition du chemin complet du fichier de sauvegarde
        torch.save(self.state_dict(), file_name)  # Sauvegarde des paramètres du modèle dans le fichier


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr  # Taux d'apprentissage
        self.gamma = gamma  # Facteur de réduction
        self.model = model  # Modèle de réseau de neurones
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)  # Initialisation de l'optimiseur Adam
        self.criterion = nn.MSELoss()  # Définition de la fonction de perte Mean Squared Error (MSE)

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)  # Conversion de l'état en tenseur
        next_state = torch.tensor(next_state, dtype=torch.float)  # Conversion de l'état suivant en tenseur
        action = torch.tensor(action, dtype=torch.long)  # Conversion de l'action en tenseur
        reward = torch.tensor(reward, dtype=torch.float)  # Conversion de la récompense en tenseur

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)  # Ajout d'une dimension pour traiter le cas d'un seul exemple
            next_state = torch.unsqueeze(next_state, 0)  # Ajout d'une dimension pour traiter le cas d'un seul exemple
            action = torch.unsqueeze(action, 0)  # Ajout d'une dimension pour traiter le cas d'un seul exemple
            reward = torch.unsqueeze(reward, 0)  # Ajout d'une dimension pour traiter le cas d'un seul exemple
            done = (done, )  # Création d'un tuple pour le marqueur d'état terminé

        pred = self.model(state)  # Prédiction des valeurs Q avec l'état actuel

        target = pred.clone()  # Création d'une copie des prédictions
        for idx in range(len(done)):  # Parcours des exemples
            Q_new = reward[idx]  # Initialisation de la nouvelle valeur Q avec la récompense
            if not done[idx]:  # Si l'épisode n'est pas terminé
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))  # Mise à jour de la valeur Q
                # Q_new = r + y * max(next_predicted Q value) -> seulement si not done
            target[idx][torch.argmax(action[idx]).item()] = Q_new  # Mise à jour de la valeur cible pour l'action choisie
    
        
        self.optimizer.zero_grad()  # Réinitialisation des gradients
        loss = self.criterion(target, pred)  # Calcul de la perte
        loss.backward()  # Rétropropagation du gradient
        self.optimizer.step()  # Mise à jour des poids du modèle



