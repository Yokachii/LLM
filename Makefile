main : perceptron.o main.o dataset.o use train
	gcc -O3 -Wall -Wextra -o perceptron perceptron.o main.o dataset.o -lm

use : perceptron.o use.o dataset.o 
	gcc -O3 -Wall -Wextra -o use perceptron.o use.o dataset.o -lm

train : perceptron.o train.o dataset.o 
	gcc -O3 -Wall -Wextra -o train perceptron.o train.o dataset.o -lm

perceptron.o : ./src/perceptron.c ./include/perceptron.h ./include/dataset.h
	gcc -c -O3 -Wall -Wextra -o perceptron.o ./src/perceptron.c

dataset.o : ./src/dataset.c ./include/dataset.h ./include/perceptron.h
	gcc -c -O3 -Wall -Wextra -o dataset.o ./src/dataset.c

main.o : ./src/main.c ./include/dataset.h ./include/perceptron.h
	gcc -c -O3 -Wall -Wextra -o main.o ./src/main.c

use.o : ./src/use.c ./include/dataset.h ./include/perceptron.h
	gcc -c -O3 -Wall -Wextra -o use.o ./src/use.c

train.o : ./src/train.c ./include/dataset.h ./include/perceptron.h
	gcc -c -O3 -Wall -Wextra -o train.o ./src/train.c

clean : 
	rm -f *.o
	rm -f perceptron
	rm -f use