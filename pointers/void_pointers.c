#include <stdio.h>

int main() {
	int x=10;
	float fx=4.5;

	void * px ; // it is void pointer, so we can have different data types assigned to them

	px = &x;
	printf("Integer values :%d\n", *(int*)px);

	px = & fx;
	printf("Float values :%.2f\n", *(float *)px);

}
