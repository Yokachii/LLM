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

Couche *new_layer(int dim, float taux, int nb_neuronnes);

void free_layer(Couche *p);

void free_layer(Couche *p);

void display_perceptron(Couche p);

void update(Reseau *n, float *tab);

void train(Reseau *n, int nb_epoques, Texte t, int *nb_epoques_max, int humanOutput);

float** forward_network(Reseau *net, float *input);

#endif