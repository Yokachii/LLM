#ifndef DATASET_H
#define DATASET_H


typedef struct dataset_s
{
    int n;
    int dim;
    float **data;
} Dataset;


typedef struct
{
    int TP;
    int TN;
    int FN;
    int FP;
} MatrixConfusion;


/**
 * \fn dataset *new_dataset(int dim, int taux) 
 * \brief Initialise un perceptron
 *
 * \param dim nombre de wi xi 
 * \param n nombre de données 
 * \return pointeur vers le nouveau perceptron
 */ 
Dataset *new_dataset(int dim, int n);

/**
 * \fn void *free_dataset(Dataset *p)
 * \brief Libère un dataset
 *
 * \param p Pointeur vers le dataset à free 
 */ 
void *free_dataset(Dataset *d);


/**
 * \fn Dataset *load_dataset(char *path)
 * \brief Charge un dataset
 *
 * \param path local path
 */ 
Dataset *load_dataset(char *path, int *V);

/**
 * \fn void *display_dataset(dataset p)
 * \brief Affiche un dataset
 *
 * \param p dataset à afficher
 */ 
void display_dataset(Dataset p);


/**
 * \fn float sigmoide(int z)
 */ 
float sigmoide(float z);



/**
 * \fn void shuffle(Dataset *d)
 * \brief Mélange les données d'un dataset
 *
 * \param d dataset à mélanger
 */ 
void shuffle(Dataset *d);


/**
 * \fn void normalize(Dataset *d)
 * \brief Normalise les données d'un dataset
 *
 * \param d dataset à normaliser
 */ 
void normalize(Dataset *d);


/**
 * \fn void split_dataset(Dataset *d, Dataset *train, Dataset *test, float ratio)
 * \brief Sépare les données en deux dataset
 *
 * \param ratio 0 -> 1,EX 0.8 = 80%
 */ 
void split_dataset(Dataset *d, Dataset **train, Dataset **test, float ratio);

#endif