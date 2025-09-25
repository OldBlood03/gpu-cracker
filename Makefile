CC = nvcc 

SOURCE_FILES = $(wildcard *.cu)
OBJ_FILES = $(SOURCE_FILES:.cu=.o)

LOPTS = -L/usr/local/lib -lssl -lcrypto
COPTS = -g -Xcompiler "-Wall -Wextra" 

all: bin

bin: $(OBJ_FILES)
	$(CC) $(OBJ_FILES) -o bin $(LOPTS)

%.o: %.cu
	$(CC) $(COPTS) -c $< -o $@ 

.PHONY: clean debug run

clean:
	rm bin $(OBJ_FILES)

debug: all
	gdb ./bin
run: all
	./bin
