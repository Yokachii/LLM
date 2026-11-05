#include <stdio.h> 


#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include "dataset.h"

typedef struct couche_s
{
    float *b;
    int dim;
    float taux;
    float **poids;
    int nb_neuronnes;
} Couche;

typedef struct reseau_s {
    int nb_layer;
    Couche **Layers;
} Reseau;



void free_network(Reseau *n);

void display_network(Reseau n);

Reseau *new_network(int nb_couche_cache, int nb_neuronnes_ch, float taux, int dimE, int dimO);

/**
 * \fn perceptron *new_perceptron(int dim, int taux) 
 * \brief Initialise un perceptron
 *
 * \param dim nombre de wi xi 
 * \param taux taux d'apprentissage 
 * \return pointeur vers le nouveau perceptron
 */ 
Couche *new_layer(int dim, float taux, int nb_neuronnes);


/**
 * \fn void free_perceptron(Perceptron *p)
 * \brief Libère un perceptron
 *
 * \param p Pointeur vers le perceptron à free 
 */ 
void free_layer(Couche *p);


/**
 * \fn void free_layer(Perceptron *p)
 * \brief Libère un perceptron
 *
 * \param p Pointeur vers le perceptron à free 
 */ 
void free_layer(Couche *p);

/**
 * \fn void *display_perceptron(Perceptron p)
 * \brief Affiche un perceptron
 *
 * \param p Perceptron à afficher
 */ 
void display_perceptron(Couche p);


/**
 * \fn int prediction(Perceptron p, int* x)
 * \brief Prediction
 *
 * \param p perceptron
 * \param x vecteur
 */ 
float *prediction(Couche p, float* x);


/**
 * \fn void update(Perceptron *p, float *tab)
 * \brief Met à jour les poids du perceptron
 *
 * \param p perceptron
 * \param tab vecteur de données
 */ 
void update(Reseau *n, float *tab);


/**
 * \fn float train(Reseau *n, int nb_epoques, Dataset *d, Dataset *dt, int breakIfTrained, int *nb_epoques_max, float *acc_max)
 * \brief Entraine le perceptron sur un dataset
 *
 * \param p perceptron
 * \param nb_epoques nombre de cycle
 * \param d Dataset to train on
 */ 
float train(Reseau *n, int nb_epoques, Dataset *d, Dataset *dt, int breakIfTrained, int *nb_epoques_max, float *acc_max, int humanOutput);


/**
 * \fn float calc_acc(Perceptron *p, Dataset *d)
 * \brief Calcule l'acc d'un perceptron sur un dataset donné
 *
 * \param p perceptron
 * \param d Dataset to test on
 */ 
float calc_acc(Reseau *n, Dataset *d);


/**
 * \fn MatrixConfusion matrice_confusion(Perceptron *p, Dataset *d)
 * \brief Calcul la matrix de confusion
 *
 * \param p perceptron
 * \param d Dataset to test on
 */ 
MatrixConfusion *matrice_confusion(Reseau *net, Dataset *d);


/**
 * \fn void display_matrix(MatrixConfusion m)
 * \brief Affiche la matrix de confusion
 *
 * \param m La matrix à afficher
 */ 
void display_matrix(MatrixConfusion *m);


float** forward_network(Reseau *net, float *input);

#endif