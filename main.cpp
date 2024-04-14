#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <fstream>

using namespace std;
using namespace std::chrono;

//Class Definition

struct LinearModel {
    double m; 
    double c; 

    //LinearModel() : m(0.0), c(0.0) {} //Construct?

    
    double predict(double x) const {
        return m * x + c;
    }

    //SGD Learning
    void updateSGD(vector<double> x_data, vector<double> y_data, double learning_rate, int epochs) {

        for (int i=0;i<epochs;i++) {
            int index = rand() % x_data.size();
            double x = x_data[index];
            double y = y_data[index];

            double y_pred = m*x + c;
            double error = y_pred - y;
            m -= learning_rate * error * x;
            c -= learning_rate * error;

        }
    }

    void pupdateSGD(std::vector<double>& x_data, std::vector<double>& y_data, double learning_rate, int epochs) {
        double m = 0.0;  // Assuming m and c are members of LinearModel
        double c = 0.0;

        omp_set_num_threads(8);

        #pragma omp parallel shared(x_data, y_data, m, c, learning_rate, epochs)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, x_data.size() - 1);

            // Use private copies of m and c for each thread
            double private_m = m;
            double private_c = c;

            #pragma omp for
            for (int i = 0; i < epochs; ++i) {
                int index = dis(gen);
                double x = x_data[index];
                double y = y_data[index];

                double y_pred = private_m * x + private_c;
                double error = y_pred - y;
                private_m -= learning_rate * error * x;
                private_c -= learning_rate * error;
            }

            // Combine results after the parallel loop
            #pragma omp critical
            {
                m += private_m;
                c += private_c;
            }
        }

        // Update the original m and c values outside the parallel region
        m /= epochs;
        c /= epochs;
    }
};


void generateData(vector<double>& x, vector<double>& y, int num_points) {

    mt19937 gen(time(0)); // Seed the random number generator with current time
    random_device rd;
    uniform_real_distribution<double> dist(0.0, 10.0);

    for (int i = 0; i < num_points; ++i) {
        double rand_x = dist(gen);
        double rand_y = 2.0 * rand_x + 1.0 + dist(gen); // y = 2x + 1 + noise
        x.push_back(rand_x);
        y.push_back(rand_y);
    }
}


int main() {
    const double learning_rate = 0.0001;

    ofstream serialFile("serial.txt");
    ofstream parallelFile("parallel_8.txt");

    vector<double> x_data, y_data;
    generateData(x_data, y_data, 1000); // 1000 for training

    vector<double> test_x, test_y;
    generateData(test_x, test_y, 10); // 10 for testing (printing)

    //cout << omp_get_max_threads() << " threads are present and going to be used" << endl << endl << endl;

    vector<long> Iterations_Vector = {100000,200000,250000,500000,750000,1000000};

    for (int i=0;i<Iterations_Vector.size();i++) {

        long num_iterations = Iterations_Vector[i];
        

        cout << "Number of iterations(N) -> " << num_iterations << endl;

        LinearModel model_serial;
        LinearModel model_parallel;

        auto start_time = high_resolution_clock::now();

        model_serial.updateSGD(x_data,y_data,learning_rate,num_iterations);

        auto end_time = high_resolution_clock::now();

        auto duration = duration_cast<nanoseconds>(end_time - start_time);

        cout << "Serial Training took " << duration.count() / 1000.0 << " ms." << endl;
        serialFile << duration.count() / 1000.0 << endl;


        start_time = high_resolution_clock::now();

        model_parallel.pupdateSGD(x_data,y_data,learning_rate,num_iterations);

        end_time = high_resolution_clock::now();

        duration = duration_cast<nanoseconds>(end_time - start_time);

        cout << "Parallel Training took " << duration.count() / 1000.0 << " ms." << endl;
        parallelFile << duration.count() / 1000.0 << endl;

        cout << endl << endl;
    }


}