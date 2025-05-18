#include <stdio.h>


int main() {
	int x=10;
	int *px = &x;
	
	printf("*px is %d\n",*px);
	printf("address of x %p\n", px);
	printf(" address of px %p\n", &px);


}
