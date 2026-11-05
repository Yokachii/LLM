cln : perceptron.o main.o dataset.o 
	gcc -O3 -Wall -Wextra -o perceptron perceptron.o main.o dataset.o -lm

perceptron.o : ./src/perceptron.c ./include/perceptron.h ./include/dataset.h
	gcc -c -O3 -Wall -Wextra -o perceptron.o ./src/perceptron.c

dataset.o : ./src/dataset.c ./include/dataset.h ./include/perceptron.h
	gcc -c -O3 -Wall -Wextra -o dataset.o ./src/dataset.c

main.o : ./src/main.c ./include/dataset.h ./include/perceptron.h
	gcc -c -O3 -Wall -Wextra -o main.o ./src/main.c

clean : 
	rm -f *.o
	rm -f perceptron