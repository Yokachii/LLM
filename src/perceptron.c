#include "../include/perceptron.h"
#include <stdlib.h>
#include <math.h>

Reseau *new_network(int nb_couche_cache, int nb_neuronnes_ch, float taux, int dimE, int dimO){
    Reseau *res = malloc(sizeof(Reseau));

    res->nb_layer=2+nb_couche_cache;
    
    Couche **layers = malloc(sizeof(Couche*)*(res->nb_layer));

    // int dimL0=dimE;
    int dimL0 = nb_neuronnes_ch;

    Couche *entree = new_layer(dimE,taux,dimL0);

    layers[0] = entree;

    for(int i = 0; i<nb_couche_cache;i++){
        int dimL = nb_neuronnes_ch;
        if(i==0) dimL=dimL0;
        Couche *l = new_layer(dimL,taux,nb_neuronnes_ch);
        layers[i+1] = l;
    }

    Couche *out = new_layer(nb_neuronnes_ch,taux,dimO);

    layers[res->nb_layer-1] = out;

    res->Layers=layers;

    return res;
}

void display_network(Reseau n){
    printf("Taux : %.2f\n",n.Layers[0]->taux);
    printf("\nCouche d'entré :\n");
    display_perceptron(*(n.Layers[0]));
    printf("\n\n");
    for(int i = 0; i<n.nb_layer-2;i++){
        printf("Couche N°%d (de %d neuronnes):\n",i+1,n.Layers[0]->nb_neuronnes);
        display_perceptron(*(n.Layers[i+1]));
        printf("\n\n");
    }
    printf("Couche de sortie :\n");
    display_perceptron(*(n.Layers[n.nb_layer-1]));
    
}

void free_network(Reseau *n){
    for(int i = 0;i<n->nb_layer;i++){
        free_layer(n->Layers[i]);
    }
    free(n);
}

/**
 * \fn int new_perceptron(int dim, int taux) 
 * \brief Initialise un perceptron
 *
 * \param dim nombre de wi xi 
 * \param n nombre de données 
 * \return pointeur vers le nouveau perceptron
 */ 
Couche *new_layer(int dim, float taux, int nb_neuronnes){
    Couche *n_perceptron = malloc(sizeof(Couche));

    n_perceptron->b = 0;
    n_perceptron->dim = dim;
    n_perceptron->taux = taux;
    n_perceptron->nb_neuronnes = nb_neuronnes;

    float *biais = malloc(sizeof(float)*nb_neuronnes);

    for(int i = 0; i<nb_neuronnes; i++){
        biais[i]=0;
    }

    n_perceptron->b=biais;


    float **poids = malloc(sizeof(float*)*nb_neuronnes);

    for(int i = 0; i<nb_neuronnes; i++){
        float *p = malloc(sizeof(float)*dim);

        float scale = sqrtf(1.0f / dim);
        for(int j = 0; j<dim; j++){
            p[j] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
        }

        poids[i]=p;
    }

    n_perceptron->poids=poids;

    return n_perceptron;
};


/**
 * \fn void free_layer(Perceptron *p)
 * \brief Libère un perceptron
 *
 * \param p Pointeur vers le perceptron à free 
 */ 
void free_layer(Couche *p){
    int nb_neuronnes = p->nb_neuronnes;
    for(int i = 0;i<nb_neuronnes;i++){
        free(p->poids[i]);
    }
    free(p->b);
    free(p);
};

/**
 * \fn void *display_perceptron(Perceptron p)
 * \brief Affiche un perceptron
 *
 * \param p Perceptron à afficher
 */ 
void display_perceptron(Couche p){
    printf("Dim : %d\n",p.dim);
    for(int j = 0; j<p.nb_neuronnes;j++){
        float b = p.b[j];
        int dim = p.dim;
        printf("Perceptron N°%d\nBiais : %.2f\n[\n    ",j+1,b);
        for(int i = 0; i<dim; i++){
            printf("%.2f, ",p.poids[j][i]);
        }
        printf("\n]\n\n");
    }
};


/**
 * \fn float* prediction(Perceptron p, float* x)
 * \brief Prediction -> calc `z = w·x + b` puis retourne `σ(z)`
 *
 * \param p perceptron
 * \param x vecteur
 */ 
float* prediction(Couche p, float* x) {
    float *outputs = malloc(sizeof(float) * p.nb_neuronnes);

    for (int j = 0; j < p.nb_neuronnes; j++) {
        float z = p.b[j];
        
        for (int i = 0; i < p.dim; i++) {
            z += x[i] * p.poids[j][i];
        }
        
        outputs[j] = sigmoide(z);
    }

    return outputs;
}


float** forward_network(Reseau *net, float *input) {
    float *current_input = input;
    float **res = malloc(sizeof(float*)*net->nb_layer);

    for (int i = 0; i < net->nb_layer; i++) {
        current_input = prediction(*(net->Layers[i]), current_input);

        res[i]=current_input;
    }

    return res; 
}

void free_forward(float **f, int s){
    for(int i = 0; i<s; i++){
        free(f[i]);
    }
    free(f);
}

static float calc_loss(Reseau *n, Dataset *d){
    float loss = 0;
    for(int i = 0; i < d->n; i++){
        float **fw = forward_network(n, d->data[i]);
        float p = fw[n->nb_layer - 1][0];
        float y = d->data[i][d->dim - 1];
        free_forward(fw, n->nb_layer);

        if(p < 1e-7f) p = 1e-7f;               // évite log(0)
        if(p > 1.0f - 1e-7f) p = 1.0f - 1e-7f;

        loss -= y * logf(p) + (1.0f - y) * logf(1.0f - p);
    }
    return loss / d->n;
}

static void set_taux(Reseau *n, float taux){
    for(int l = 0; l < n->nb_layer; l++) n->Layers[l]->taux = taux;
}

/**
 * \fn void update(Perceptron *p, float *tab)
 * \brief Met à jour les poids du reseau
 *
 * \param p perceptron
 * \param tab vecteur de données
 */ 
void update(Reseau *n, float *tab) {
    int nb_layers = n->nb_layer;
    
    float **forward_res = forward_network(n, tab);
    
    float label = tab[n->Layers[0]->dim]; // Label de sortie

    float **deltas = malloc(sizeof(float*) * nb_layers);

    //Calcules sortie
    int last_idx = nb_layers - 1;
    Couche *last_layer = n->Layers[last_idx];
    deltas[last_idx] = malloc(sizeof(float) * last_layer->nb_neuronnes);

    for (int j = 0; j < last_layer->nb_neuronnes; j++) {
        float y_pred = forward_res[last_idx][j];
        
        // deltas[last_idx][j] = (y_pred - label) * y_pred * (1 - y_pred); <- problème, si le resaux se trompe complet, l'apprentissage arrète presque completement
        deltas[last_idx][j] = (y_pred - label);
    }

    for (int l = nb_layers - 2; l >= 0; l--) {
        Couche *current_layer = n->Layers[l];
        Couche *next_layer = n->Layers[l + 1];
        
        deltas[l] = malloc(sizeof(float) * current_layer->nb_neuronnes);

        for (int j = 0; j < current_layer->nb_neuronnes; j++) {
            float error_sum = 0.0f;
            
            for (int k = 0; k < next_layer->nb_neuronnes; k++) {
                error_sum += next_layer->poids[k][j] * deltas[l + 1][k];  // Formule du calcule de l'erreur
            }
            
            float r_j = forward_res[l][j];

            deltas[l][j] = error_sum * r_j * (1.0f - r_j);
        }
    }
    
    for (int l = 0; l < nb_layers; l++) {
        Couche *layer = n->Layers[l];
        float lr = layer->taux;
        float *input_l = (l == 0) ? tab : forward_res[l - 1]; // Entrées de la couche

        for (int j = 0; j < layer->nb_neuronnes; j++) {
            float delta = deltas[l][j];

            // Mise à jour des poids
            for (int i = 0; i < layer->dim; i++) {
                layer->poids[j][i] -= lr * delta * input_l[i];
            }

            
            layer->b[j] -= lr * delta;// Mise à jour du biais
        }
    }

    for (int l = 0; l < nb_layers; l++) {
        free(deltas[l]);
    }
    free(deltas);
    
    free_forward(forward_res, nb_layers);
}


/**
 * \fn float train(Reseau *n, int nb_epoques, Dataset *d, Dataset *dt, int breakIfTrained, int *nb_epoques_max, float *acc_max)
 * \brief Entraine le perceptron sur un dataset
 *
 * \param p perceptron
 * \param nb_epoques nombre de cycle
 * \param d Dataset to train on
 */ 
float train(Reseau *n, int nb_epoques, Dataset *d, Dataset *dt, int breakIfTrained, int *nb_epoques_max, float *acc_max, int humanOutput){
    int bestAccEp = 0;
    float bestAcc = -1;

    float taux0 = n->Layers[0]->taux;
    float k = (nb_epoques > 0) ? 3.0f / nb_epoques : 0.0f; // décroissance du taux

    int i;
    for(i = 0; i < nb_epoques; i++){
        set_taux(n, taux0 / (1.0f + k * i));
        shuffle(d);

        float acc = calc_acc(n, dt);

        if(acc > bestAcc){
            bestAcc = acc;
            bestAccEp = i;
        }

        if(humanOutput){
            if(i % 100 == 0){
                printf("Epoque : %d ; Acc : %.4f ; Loss : %.4f ; taux : %.4f\n",
                       i, acc, calc_loss(n, dt), n->Layers[0]->taux);
            }
        }else if(i % 10 == 0){
            // #EP;epoque;taux;acc;accMax;epoqueMax;TP;FP;TN;FN
            MatrixConfusion *m = matrice_confusion(n, dt);
            printf("#EP;%d;%.5f;%.4f;%.4f;%d;%d;%d;%d;%d\n",
                   i, n->Layers[0]->taux, acc, bestAcc, bestAccEp,
                   m->TP, m->FP, m->TN, m->FN);
            free(m);
        }

        if(breakIfTrained && acc == 1){
            break;
        }

        for(int j = 0; j < d->n; j++){
            update(n, d->data[j]);
        }
    }

    set_taux(n, taux0);
    *acc_max = bestAcc;
    *nb_epoques_max = bestAccEp;

    return i;   // nombre d'époques réellement exécutées
}



/**
 * \fn float calc_acc(Reseau *net, Dataset *d)
 * \brief Calcule l'acc d'un reseau sur un dataset donné
 *
 * \param n Reseau
 * \param d Dataset to test on
 */ 
float calc_acc(Reseau *n, Dataset *d){
    int nb_data = d->n;

    if (d->dim - 1 != n->Layers[0]->dim) {
        printf("Dimension missmatch, dataset dim : %d ; dim de la couche 1 : %d ",d->dim,n->Layers[0]->dim);
        return -1;
    }

    float res = 0;

    for(int i = 0; i < nb_data; i++){
        float **y_array_sq = forward_network(n, d->data[i]);
        float *y_array = y_array_sq[n->nb_layer-1];
        
        Couche *last_layer = n->Layers[n->nb_layer - 1];
        float y = y_array[last_layer->nb_neuronnes - 1]; 
        
        float label = d->data[i][d->dim - 1];
        
        if((y >= 0.5 && label >= 0.5) || (y < 0.5 && label < 0.5)) {
            res += 1;
        }        
        
        free_forward(y_array_sq,n->nb_layer);
    }

    return res / nb_data;
}



MatrixConfusion *matrice_confusion(Reseau *net, Dataset *d){
    MatrixConfusion *res = calloc(1, sizeof(MatrixConfusion));
    for(int i = 0; i < d->n; i++){
        float **fw = forward_network(net, d->data[i]);
        float y = fw[net->nb_layer - 1][0];
        float label = d->data[i][d->dim - 1];
        free_forward(fw, net->nb_layer);

        if(y >= 0.5 && label >= 0.5) res->TP++;
        else if(y >= 0.5)            res->FP++;
        else if(label < 0.5)         res->TN++;
        else                         res->FN++;
    }
    return res;
}

/**
 * \fn void display_matrix(MatrixConfusion m)
 * \brief Affiche la matrix de confusion
 *
 * \param m La matrix à afficher
 */ 
void display_matrix(MatrixConfusion *m){

    printf("FP : %d\nFN : %d\nTP : %d\nTN : %d\n\n",m->FP,m->FN,m->TP,m->TN);
    
}