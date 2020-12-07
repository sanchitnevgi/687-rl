#include "QLearning.hpp"

using namespace std;
using namespace Eigen;

/*
* In this assignment, the constructor takes the environment name and
* uses it to set hyperparameters like the step size, gamma, and any other parameters.
*/
QLearning::QLearning(const int& stateDim, const int& numActions, const std::string& envName)
{
    this->numActions = numActions;
    this->stateDim = stateDim;

    epsilon = 0.95;
    initPhi = false;

    // In the constructor, use envName to set hyperparameters.
	// No not use envName to initialize the policy!
	// I'm leaving in some of my code to show how you might structure this.
	if (envName.compare("Mountain Car") == 0)
	{
        gamma = 0.99;
        alpha = 0.005;

        basis = 2;
        useFourier = true;
	}
	else if (envName.compare("Cart Pole") == 0)
	{
        gamma = 0.99;
        alpha = 0.00038;

        basis = 4;
        useFourier = true;
    }
	else if (envName.compare("Gridworld") == 0)
	{
        gamma = 0.999;
        alpha = 0.34;

        useFourier = false;
	}
	else
	{
		cout << "Error: Unknown environment name in Sarsa constructor." << endl;
		exit(1);
	}

    if (useFourier)
    {
        numFeatures = ipow(basis + 1, stateDim);

        phi = VectorXd::Zero(numFeatures);
        w = MatrixXd::Zero(numActions, numFeatures);
    }
    else
    {
        q = MatrixXd::Zero(stateDim, numActions);
    }
}

bool QLearning::updateBeforeNextAction()
{
	return true;
}

// Epsilon-greedy or Softmax action selection
int QLearning::getAction(const Eigen::VectorXd& s, std::mt19937_64& generator)
{
    VectorXd actionProbabilities;
    if (useFourier)
    {
        VectorXd phiPrime = getFeatures(s);
        actionProbabilities = (w * phiPrime);
    }
    else
    {
        int currState = convertOneHotToInt(s);
        actionProbabilities = q.row(currState);
    }

    actionProbabilities = actionProbabilities.array().exp();
    actionProbabilities /= actionProbabilities.sum();

    int action = softmaxActionSelection(actionProbabilities, generator);

    return action;
}

// Tell the agent that it is at the start of a new episode
void QLearning::newEpisode()
{
	initPhi = false;
}

// Update given a (s,a,r,s') tuple
void QLearning::update(const Eigen::VectorXd& s, const int& a, const double& r, const Eigen::VectorXd& sPrime, std::mt19937_64& generator)
{
    if (useFourier)
    {
        if (!initPhi) {
            phi = getFeatures(s);
            initPhi = true;
        }

        VectorXd phiPrime = getFeatures(sPrime);

        // TODO
        int aPrime = 0;

        double delta = r + gamma * (r + w.row(aPrime).dot(phiPrime) - w.row(a).dot(phi));

        for (int f = 0; f < numFeatures; ++f) {
            w(a, f) += alpha * delta * phi[f];
        }

        phi = phiPrime;
    }
    else
    {
        int currState = convertOneHotToInt(s);
        int nextState = convertOneHotToInt(sPrime);

        // TODO
        int aPrime = 0;

        double delta = r + gamma * q(nextState, aPrime) - q(currState, a);
        q(currState, a) += alpha * delta;
    }
}

// Let the agent update/learn when sPrime would be the terminal absorbing state
void QLearning::update(const Eigen::VectorXd& s, const int& a, const double& r, mt19937_64& generator)
{
	// @TODO: Fill in this function.
}

int QLearning::softmaxActionSelection(VectorXd probabilities, std::mt19937_64& generator)
{
    double value = uniform_real_distribution<double>(0, 1)(generator);
    double runningSum = 0;

    for (int action = 0; action < (int) probabilities.size(); ++action) {
        runningSum += probabilities[action];
        if (value <= runningSum) {
            return action;
        }
    }

    return numActions - 1;
}

/**
 * Build a feature vector using fourier basis
 * */
VectorXd QLearning::getFeatures(const Eigen::VectorXd& s)
{
    VectorXd nextPhi = VectorXd::Zero(numFeatures);
    VectorXd c = VectorXd::Zero(stateDim);

    for (int f = 0; f < numFeatures; ++f) {
        c.fill(f);
        nextPhi[f] = cos(M_PI * s.dot(c));
    }
    return nextPhi;
}

int QLearning::convertOneHotToInt(VectorXd s)
{
    for (int state; state < (int) s.size(); ++state)
        if (s[state] != 0)
            return state;
    return 0;
}

/*
* My code used this function. It returns a^b, where a and b are both integers.
* You may not need this, but if you do, we are including it.
*/
int QLearning::ipow(const int& a, const int& b) {
	if (b == 0) return 1;
	if (b == 1) return a;
	int tmp = ipow(a, b / 2);
	if (b % 2 == 0) return tmp * tmp;
	else return a * tmp * tmp;
}

/*
* This is another function that my code uses, and which you are welcome to use.
* It is entirely possible that your code won't require this function.
* This function takes as input a vector, buff, that represents a number in
* base (maxDigit+1), and adds one to the counter. Overflow behavior is not defined.
*/
void QLearning::incrementCounter(VectorXd& buff, const int& maxDigit) {
	for (int i = 0; i < (int)buff.size(); i++) {
		buff[i]++;
		if (buff[i] <= maxDigit)
			break;
		buff[i] = 0;
	}
}