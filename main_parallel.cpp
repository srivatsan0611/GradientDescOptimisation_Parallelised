#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <omp.h>


using namespace std;
using namespace std::chrono;


struct LinearModel {
    double m; 
    double c; 

    LinearModel() : m(0.0), c(0.0) {} //Construct?

    
    double predict(double x) const {
        return m * x + c;
    }

    //SGD Learning
    void updateSGD(double x, double y, double learning_rate) {
        double y_pred = predict(x);
        double error = y_pred - y;
        m -= learning_rate * error * x;
        c -= learning_rate * error;
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


    const int num_iterations = 5000;
    const double learning_rate = 0.01;

    vector<double> x_data, y_data;
    generateData(x_data, y_data, 1000); // 1000 for training

    vector<double> test_x, test_y;
    generateData(test_x, test_y, 10); // 10 for testing (printing)

    LinearModel model;


    //Training

    
    auto start_time = high_resolution_clock::now(); // Start timing


    #pragma omp parallel for shared(x_data, y_data, model) default(none) schedule(static) num_threads(omp_get_max_threads())
    for (int i = 0; i < num_iterations; ++i) {

        //Start with random index
        int index = rand() % x_data.size();
        double x = x_data[index];
        double y = y_data[index];


        #pragma omp critical
        model.updateSGD(x, y, learning_rate);

        /* if (i % 1000 == 0) {
            cout << "Iteration " << i << ": m = " << model.m << ", c = " << model.c << endl;
        } */
    }

    cout << "Final model: y = " << model.m << "x + " << model.c << endl;


    auto end_time = high_resolution_clock::now(); // End timing
    auto duration = duration_cast<nanoseconds>(end_time - start_time);

    cout << "Training took " << duration.count() << " nanoseconds." << endl;


    
    cout << "\nTesting:\n" << endl;
    for (size_t i = 0; i < test_x.size(); ++i) {
        double x = test_x[i];
        double y_true = test_y[i];
        double y_pred = model.predict(x);
        cout << "True y = " << y_true << ", Predicted y = " << y_pred << endl;
    }

    return 0;
}
