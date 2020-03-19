// perform 2d convolution. Output channel 2^i (i=0,1,...,10)

#include <stdio.h>
#include <time.h>
#include <math.h>


int c_conv(int in_channel, int o_channel, int kernel_size, int stride)
{
    int height = 1280;
    int width = 720;
    size_t nbytes = height * width * in_channel * sizeof(char);
    int i, j, k, m, n, p;
    double val;
    long int row;
    int num = 0;

    // simulate input image as all black
    char *input_img = (char*)malloc(nbytes);
    memset(input_img, 0, nbytes);

    char *out_img = (char *)malloc((height - 1) * (width - 1) * sizeof(char));
    memset(out_img, 0, (height - 1) * (width - 1) * sizeof(char));

    double *k1 = (double*)malloc(kernel_size*kernel_size*sizeof(double));

    for (m = 0; m < o_channel; m++)
    {
        // randomly generate kernel
        for (p = 0; p<kernel_size*kernel_size; p++)
                k1[p] = rand() / (2.0) - 1;

        for (i = 1; i < height - 1; i++)
            for (j = 1; j < width - 1; j++)
            {
                val = 0.0;
                row = i * width + j;

                for (n = 0; n < in_channel; n++)
                {
                    val += input_img[(row - width - 1) * in_channel + n] * k1[0] + input_img[(row - width) * in_channel + n] * k1[1] + input_img[(row - width + 1) * in_channel + n] * k1[2] +
                        input_img[(row - 1) * in_channel + n] * k1[3] + input_img[(row) * in_channel + n] * k1[4] + input_img[(row + 1) * in_channel + n] * k1[5] +
                        input_img[(row + width - 1) * in_channel + n] * k1[6] + input_img[(row + width) * in_channel + n] * k1[7] + input_img[(row + width + 1) * in_channel + n] * k1[8];
                }
                out_img[(i - 1)*(width - 1) + j - 1] = val / (double)in_channel;
                num += (kernel_size * kernel_size * 2 - 1) * in_channel;
            }
    }
    return num;
}


void main()
{
	int o_channel[11]; // output channels
	clock_t t;
	double times[11];
	double val; // value storing convolution results
  int in_channel = 3, kernel_size = 3, stride = 1;
  int numOperates;
  int num;
  int k;


	for (k = 0; k < 11; k++)
	{
		o_channel[k] = pow(2, k); // # of output channels
    printf("%d:  ", o_channel[k]);
    numOperates = 0;
		
		// do 2d convolution 
		t = clock();
    num = c_conv(in_channel, o_channel[k], kernel_size, stride);
    numOperates += num;


		times[k] = (clock() - t) / (double)CLOCKS_PER_SEC;

		printf("%f\n", times[k]);
	}

}

