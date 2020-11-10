#include "CartPole.hpp"

using namespace std;
using namespace Eigen;

CartPole::CartPole(mt19937_64 & generator)
{
	newEpisode(generator);
}

int CartPole::getMaxEps() const
{
	return 1000;
}

int CartPole::getStateDim() const
{
	return 4;
}

int CartPole::getNumActions() const
{
	return 2;
}

double CartPole::getGamma() const
{
	return 1.0;
}

double CartPole::transition(const int& a, mt19937_64 & generator)
{
	double F = a * uMax + (a - 1) * uMax, omegaDot, vDot, subDt = dt / (double)simSteps;
	for (int i = 0; i < simSteps; i++) {
		omegaDot = (g * sin(theta) + cos(theta) * (muc * sign(v) - F - m * l * omega * omega * sin(theta)) / (m + mc) - mup * omega / (m * l)) / (l * (4.0 / 3.0 - m / (m + mc) * cos(theta) * cos(theta)));
		vDot = (F + m * l * (omega * omega * sin(theta) - omegaDot * cos(theta)) - muc * sign(v)) / (m + mc);
		theta += subDt * omega;
		omega += subDt * omegaDot;
		x += subDt * v;
		v += subDt * vDot;
		theta = wrapPosNegPI(theta);
		t += subDt;
	}
	x = bound(x, xMin, xMax);
	v = bound(v, vMin, vMax);
	theta = bound(theta, thetaMin, thetaMax);
	omega = bound(omega, omegaMin, omegaMax);
	return 1;
}

VectorXd CartPole::getState() const
{
	VectorXd result(4);
	result[0] = normalize(x, xMin, xMax);
	result[1] = normalize(v, vMin, vMax);
	result[2] = normalize(theta, thetaMin, thetaMax);
	result[3] = normalize(omega, omegaMin, omegaMax);
	return result;
}

bool CartPole::inTAS() const
{
	return ((fabs(theta) > M_PI / 15.0) || (fabs(x) >= 2.4) || (t >= 20.0 + 10 * dt));
}

void CartPole::newEpisode(mt19937_64 & generator)
{
	theta = omega = v = x = t = 0;
}

/*
Floating-point modulo:
The result (the remainder) has same sign as the divisor.
Similar to matlab's mod(); Not similar to fmod() because:
Mod(-3,4)= 1
fmod(-3,4)= -3
*/
double CartPole::Mod(const double& x, const double& y) {
	if (0. == y) return x;
	double m = x - y * std::floor(x / y);
	// handle boundary cases resulted from floating-point cut off:
	if (y > 0) {
		if (m >= y)
			return 0;
		if (m < 0) {
			if (y + m == y) return 0;
			else return (y + m);
		}
	}
	else
	{
		if (m <= y) return 0;
		if (m > 0) {
			if (y + m == y) return 0;
			else return (y + m);
		}
	}
	return m;
}

// wrap [rad] angle to [-PI..PI)
double CartPole::wrapPosNegPI(const double& theta)
{
	return Mod((double)theta + M_PI, (double)2.0 * M_PI) - (double)M_PI;
}

double CartPole::sign(const double& x)
{
	return (x > 0) - (x < 0);
}

double CartPole::bound(const double& x, const double& minValue, const double& maxValue) 
{
	return min(maxValue, max(minValue, x));
}

double CartPole::normalize(const double& x, const double& minValue, const double& maxValue) 
{
	double temp = bound(x, minValue, maxValue);
	return (x - minValue) / (maxValue - minValue);
}