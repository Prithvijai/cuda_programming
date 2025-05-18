#include <stdio.h>

typedef struct {
	float x;
	float y;

} Point;

int main() {
	Point p = {1.2, 3.4};
	printf("Size of Point p is :%ld\n",sizeof(p));


}
