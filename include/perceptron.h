#include <stdio.h> 


#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include "dataset.h"

int *get_indices(Modele *m, unsigned char *in, int size);

Modele *new_modele(int n, Vocab *voc, Reseau *net);

void free_modele(Modele *m);

void free_network(Reseau *n);

void display_network(Reseau n);

Reseau *new_network(int nb_couche_cache, int nb_neuronnes_ch, float taux, int dimE, int dimO);

Couche *new_layer(int dim, float taux, int nb_neuronnes, int act);

void free_layer(Couche *p);

void free_layer(Couche *p);

void display_perceptron(Couche p);

void generer(Modele *m, int longueur, float temperature, const int *amorce);

void train_modele(Modele *m, const Texte *train, const Texte *val, int epoques, float taux0);

float calc_perte(Modele *m, const Texte *t, int pas);

void update_modele(Modele *m, int *ctx, int cible);

#endif