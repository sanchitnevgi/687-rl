#include "Sarsa.hpp"

using namespace std;
using namespace Eigen;

/*
* In this assignment, the constructor takes the environment name and
* uses it to set hyperparameters like the step size, gamma, and any other parameters.
*/
Sarsa::Sarsa(const int& stateDim, const int& numActions, const std::string & envName)
{
	// @TODO: Fill in this function
	this->numStates = stateDim;
	this->numActions = numActions;
	int dOrder = 1;
	this->env = 0;
	// In the constructor, use envName to set hyperparameters.
	// No not use envName to initialize the policy!
	// I'm leaving in some of my code to show how you might structure this.
	if (envName.compare("Mountain Car") == 0)
	{
		// @TODO: Set hyperparameters for Mountain Car.
		this->gamma = 0.99;
		this->alpha = 0.005;
		this->epsilon = 1;
		dOrder = 2;
	}
	else if (envName.compare("Cart Pole") == 0)
	{
		// @TODO: Set hyperparameters for Cart Pole.
		this->gamma = 0.99;
		this->alpha = 0.00039;
		//this->alpha = 0.0002;
		this->epsilon = 1;
		//dOrder = 4;
		dOrder = 4;

	}
	else if (envName.compare("Gridworld") == 0)
	{
		// @TODO: Set hyperparameters for Gridworld.
		this->gamma = 0.999;
		this->alpha = 0.4;
		this->epsilon = 1;
		dOrder = 1;
		this->env = 1;
	}
	else
	{
		cout << "Error: Unknown environment name in Sarsa constructor." << endl;
		exit(1);
	}
	
	// @TODO: Fill in the remainder of this function.
	if (env == 0) {
		this->nTerms = ipow(dOrder+1, numStates);
		//this->nTerms = nTerms + (3*numStates);
		c = MatrixXd::Zero(nTerms, numStates);

		w = MatrixXd::Zero(numActions, nTerms);
		VectorXd counter = VectorXd::Zero(numStates);
		int termCount = 0;
		for (; termCount < nTerms; termCount++) {				// First add the dependent terms
			c.row(termCount) = counter;
			incrementCounter(counter, dOrder);
		}

		phi = phiPrime = VectorXd::Zero(nTerms);
	}
	else {
		curQ = MatrixXd::Zero(numStates, numActions);
	}
}

bool Sarsa::updateBeforeNextAction()
{
	return false;
}

// Epsilon-greedy or Softmax action selection
int Sarsa::getAction(const Eigen::VectorXd& s, std::mt19937_64& generator)
{
	// @TODO: Fill in this function.
	VectorXd actionProbabilities;
	if (env == 1) {
		int state;
		for (state = 0; state < (int)s.size(); state++)
			if (s[state] != 0)
				break;
		assert(state != (int)s.size());
		actionProbabilities = (curQ.row(state) * epsilon).array() .exp() ;
	}
	else {
		VectorXd features = cosine_calculation(s);
		actionProbabilities = (w * features * epsilon).array().exp() ;
	}
	
	actionProbabilities.array() /= actionProbabilities.sum();

	double temp = uniform_real_distribution<double>(0, 1)(generator), sum = 0;
	for (int a = 0; a < numActions; a++)
	{
		sum += actionProbabilities[a];
		if (temp <= sum)
			return a;
	}
	return numActions - 1;
}

// Tell the agent that it is at the start of a new episode
void Sarsa::newEpisode()
{
	// @TODO: Fill in this function.
	phiInit = false;
}

// Update given a (s,a,r,s') tuple
void Sarsa::update(const Eigen::VectorXd& s, const int& a, const double& r, const Eigen::VectorXd& sPrime, const int & aPrime, std::mt19937_64& generator)
{
	// @TODO: Fill in this function.
	if (env == 1) {
		int state;
		for (state = 0; state < (int)s.size(); state++)
			if (s[state] != 0)
				break;
		assert(state != (int)s.size());

		int statePrime;
		for (statePrime = 0; statePrime < (int)sPrime.size(); statePrime++)
			if (sPrime[statePrime] != 0)
				break;
		assert(statePrime != (int)sPrime.size());

		double delta = r + gamma * curQ(statePrime, aPrime) - curQ(state, a);
		curQ(state, a) += alpha * delta;
	}
	else {
		if (!phiInit) {
			phi = cosine_calculation(s);
			phiInit = true;
		}

		phiPrime = cosine_calculation(sPrime);
		double delta = r + (gamma * w.row(aPrime).dot(phiPrime)) - w.row(a).dot(phi);

		for (int i = 0; i < nTerms; i++)
			w(a, i) = w(a, i) + alpha * delta * phi[i];

		phi = phiPrime;
	}

}

// Let the agent update/learn when sPrime would be the terminal absorbing state
void Sarsa::update(const Eigen::VectorXd& s, const int& a, const double& r, mt19937_64 & generator)
{
	// @TODO: Fill in this function.
	if (env == 1) {
		int state;
		for (state = 0; state < (int)s.size(); state++)
			if (s[state] != 0)
				break;
		assert(state != (int)s.size());

		double delta = r - curQ(state, a);
		curQ(state, a) += alpha * delta;
	}
	else {
		double delta = r - w.row(a).dot(phi);

		for (int i = 0; i < nTerms; i++)
			w(a, i) = w(a, i) + alpha * delta * phi[i];
	}
	
}

/*
* My code used this function. It returns a^b, where a and b are both integers.
* You may not need this, but if you do, we are including it.
*/
int Sarsa::ipow(const int& a, const int& b) {
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
void Sarsa::incrementCounter(VectorXd & buff, const int& maxDigit) {
	for (int i = 0; i < (int)buff.size(); i++) {
		buff[i]++;
		if (buff[i] <= maxDigit)
			break;
		buff[i] = 0;
	}
}

VectorXd Sarsa::cosine_calculation(const VectorXd& x) {
	VectorXd result(nTerms);
	for (int i = 0; i < nTerms; i++) {
		result[i] = cos(M_PI * c.row(i).dot(x));
	}
	return result;
}