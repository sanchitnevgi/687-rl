#pragma once

#define _USE_MATH_DEFINES 
#include <math.h>

#include "Environment.hpp"

class CartPole : public Environment
{
public:
	CartPole(std::mt19937_64 & generator);

	int getMaxEps() const override;
	int getStateDim() const override;
	int getNumActions() const override;
	double getGamma() const override;
	double transition(const int& a, std::mt19937_64& generator) override;
	Eigen::VectorXd getState() const override;
	bool inTAS() const override;
	void newEpisode(std::mt19937_64& generator) override;

private:
	// Standard parameters for the CartPole domain
	const int simSteps = 10;
	const double dt = 0.02;
	const double uMax = 10.0;
	const double l = 0.5;
	const double g = 9.8;
	const double m = 0.1;
	const double mc = 1;
	const double muc = 0.0005;
	const double mup = 0.000002;

	// State variables ranges
	const double xMin = -2.4;
	const double xMax = 2.4;
	const double vMin = -10;
	const double vMax = 10;
	const double thetaMin = -M_PI / 12.0;
	const double thetaMax = M_PI / 12.0;
	const double omegaMin = -M_PI;
	const double omegaMax = M_PI;

	// State variables
	double x;
	double v;
	double theta;
	double omega;
	double t;


	static double Mod(const double& x, const double& y);
	static double wrapPosNegPI(const double& theta);
	static double sign(const double& x);
	static double bound(const double& x, const double& minValue, const double& maxValue);
	static double normalize(const double& x, const double& minValue, const double& maxValue);
};