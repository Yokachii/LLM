#ifndef DATASET_H
#define DATASET_H

typedef struct {
    int V;                       // nombre de caractères distincts
    int d;                       // taille du vecteur de chaque caractère
    int *dic;                // caractère -> indice, -1 si absent
    unsigned char *dic_inv;  // indice -> caractère
    float **emb;                 // emb[indice][j] : V lignes de d floats (appris)
} Vocab; // <- table d'embeding + vocab

typedef struct {
    int n;                      // nombre de tokens
    unsigned char *tok;         // n indices <- contient des indices de caractères (0..V-1)
    Vocab *voc; // Lien vers le vocabulaire
} Texte;

Texte *new_text(int n, Vocab *voc);

void free_text(Texte *t);
void free_vocab(Vocab *v);

Texte *load_texte(char *path, Vocab **v, int d);


void display_Text(Texte t);
void display_Vocab(Vocab v);


float sigmoide(float z);

void split_texte(Texte *t, Texte **train, Texte **test, float ratio);

#endif