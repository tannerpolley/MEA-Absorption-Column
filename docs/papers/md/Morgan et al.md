\title{
Development of a framework for sequential Bayesian design of experiments: Application to a pilot-scale solvent-based $\mathrm{CO}_{2}$ capture process
}

\author{
Joshua C. Morgan ${ }^{\mathrm{a}, \mathrm{b}}$, Anderson Soares Chinen ${ }^{\mathrm{a}, \mathrm{c}}$, Christine Anderson-Cook ${ }^{\mathrm{d}}$, Charles Tong ${ }^{\mathrm{e}}$, John Carroll ${ }^{\mathrm{f}}$, Chiranjib Saha ${ }^{\mathrm{f}}$, Benjamin Omell ${ }^{\mathrm{b}}$, Debangsu Bhattacharyya ${ }^{\mathrm{a}, *}$, Michael Matuszewski ${ }^{\mathrm{b}}$, K. Sham Bhat ${ }^{\mathrm{d}}$, David C. Miller ${ }^{\mathrm{b}}$ \\ ${ }^{\mathrm{a}}$ Dept. of Chemical and Biomedical Engineering, West Virginia University, Morgantown, WV 26506, United States \\ ${ }^{\mathrm{b}}$ National Energy Technology Laboratory, Pittsburgh, PA 15236, United States \\ ${ }^{\mathrm{c}}$ National Energy Technology Laboratory, Morgantown, WV 26507, United States \\ ${ }^{\mathrm{d}}$ Los Alamos National Laboratory, Los Alamos, NM 87545, United States \\ ${ }^{\mathrm{e}}$ Lawrence Livermore National Laboratory, Livermore, CA 94550, United States \\ ${ }^{\mathrm{f}}$ National Carbon Capture Center, Wilsonville, AL 35186, United States
}

\section*{H I G H L I G H T S}
- Developed a methodology for sequential Bayesian design of experiments.
- Minimized the maximum model prediction uncertainty for key output variables.
- Methodology applied to an aqueous monoethanolamine pilot plant.
- Two iteration resulted in $50 \%$ reduction in uncertainty of $\mathrm{CO}_{2}$ capture prediction.
- Methodology is generic and can be readily applied to other process systems.

\section*{ARTICLE INFO}

\section*{Keywords:}

Design of experiment
Bayesian
Sequential
Pilot plant
$\mathrm{CO}_{2}$ capture
MEA

\begin{abstract}
In this paper, a methodology is developed for sequential design of experiments (SDoE) for process systems and applied to a solvent-based $\mathrm{CO}_{2}$ capture system. In this approach, the prior knowledge of the system is used to prioritize process data collection at specific operating conditions. These data are then incorporated into a Bayesian inference methodology for updating a stochastic model by refining estimations of its underlying parameters, and the updated model is then used to generate the next set of test runs. Thus, the new knowledge obtained from the data is used to guide subsequent iterations of the experimental runs, ensuring that the overall data collection is maximally informative given that most experimental campaigns, especially at pilot or higherscale plants, are costly, time-consuming, and resource-limited. The test run objective for this work was to minimize the maximum model prediction uncertainty for key output variables, but the methodology is generic and can be readily applied to other test run objectives. This methodology is applied to an aqueous monoethanolamine (MEA) pilot plant campaign at the National Carbon Capture Center (NCCC) in Wilsonville, Alabama, USA. The SDoE framework was utilized for two iterations, while collecting 18 sets of data representing different process conditions, and this resulted in an overall average reduction in uncertainty of approximately $50 \%$ in the prediction of $\mathrm{CO}_{2}$ capture percentage. Moreover, 11 additional data sets were obtained with variation of absorber packing height for further model validation. This work shows the capability of the SDoE framework to maximize learning given limited resources, allowing for the reduction of model uncertainty, which is of great importance for many applications including reduction of technical risk associated with scale-up and economic analysis.
\end{abstract}

\footnotetext{
*Corresponding author.
E-mail address: Debangsu.Bhattacharyya@mail.wvu.edu (D. Bhattacharyya).
}

\section*{1. Introduction}

Technologies for $\mathrm{CO}_{2}$ capture and storage (CCS) are widely considered to be necessary for mitigating the effects of climate change [1]. However, there are considerable costs associated with such technologies. For instance, it has been estimated that the installation of the most mature technologies for CCS into an existing coal-fired power plant could result in a reduction of power output by up to $33 \%$, or $20 \%$ for a more advanced power plant (e.g. integrated gasification combined cycle) [2]. Naturally, these challenges have inspired many research initiatives for characterization of existing and development and demonstration of new CCS technologies, from both computational and experimental approaches. In one such long-term computational effort, the United States Department of Energy's Carbon Capture Simulation Initiative (CCSI) [3] was established in 2011 as a consortium of national laboratories, academic institutions, and industrial partners with the goal of developing a suite of computational tools and models for accelerating the development and commercialization of $\mathrm{CO}_{2}$ capture processes. The cumulative works of the CCSI program are now available to the public in the open source CCSI Toolset (located at https://github. com/CCSI-Toolset). In 2017, a successor project called Carbon Capture Simulation for Industry Impact (CCSI ${ }^{2}$ ) was initiated with the focus of utilizing the existing CCSI Toolset to support the development and scale up of several $\mathrm{CO}_{2}$ capture technologies while also expanding its capabilities to maximize learning by integrating experimental testing with models.

Development and scale-up of process technologies require accurate models for such purposes as optimization and economic analysis which are necessary for determining feasibility of technologies as well as comparing alternatives. However, predictions made from process systems models naturally contain some level of epistemic uncertainty, or uncertainty related to incomplete knowledge, inherited from underlying sub-models that are necessary for characterization of the complete process [4,5]. In chemical process systems, including those used for $\mathrm{CO}_{2}$ capture technologies, these sources of uncertainty include, but are not limited to, equipment performance, thermodynamic and transport properties of the system of interest, and its mass transfer and reaction kinetics. Uncertainty exists in both the form of the models and their parameters, such as those characterizing the physical properties and equipment design and performance, as well as in the experimental data used in model development [6].

Previous work has noted that uncertainty quantification (UQ) in chemical process design and analysis is often neglected due to the computational expense of its implementation and the scarcity of data required to appropriately characterize the uncertainty. In 1996, Whiting noted that the characterization of experimental data uncertainty, an essential requirement for quantifying the uncertainty of corresponding models, is often incomplete or missing for published data [6]. In a later (2014) paper [7], Mathias notes efforts, including strict requirements adapted by various journals, to improve reporting of uncertainty in thermodynamic data, which are key for accurate modeling of process systems. In the same paper, Mathias also notes that rigorous uncertainty analysis is still not common industrial practice for chemical process systems and attributes this partly to the difficulty of implementing uncertainty analysis in commercial process simulators. However, the inclusion of UQ in chemical engineering research has recently become more common with advances in computational techniques such as response surface methodologies (RSM), in which rigorous process models are replaced by accurate reduced-order surrogates [5,8].

Due to the importance of incorporating UQ into the modeling of $\mathrm{CO}_{2}$ capture systems, a stochastic modeling framework was developed, as part of the CCSI project, in which epistemic uncertainty is quantified at the submodel level and propagated through the full process model in order to analyze the uncertainty in key process variables such as $\mathrm{CO}_{2}$ capture performance and energy requirements for solvent regeneration.

In several papers published between 2015 and 2018, this framework was applied to an aqueous monoethanolamine (MEA) system, in which stochastic sub-models are developed for standalone property models (viscosity, molar volume, surface tension) [9], a thermodynamic framework incorporating data from multiple sources (vapor-liquid equilibrium (VLE), heat capacity, heat of absorption) [10], and process dependent models, namely mass transfer, interfacial area, and hydraulics [11]. These submodels were combined into a complete process model of an existing MEA-based $\mathrm{CO}_{2}$ capture unit with submodel uncertainty propagated through the full model, enabling estimation of uncertainty in key output variables [12].

A recent (2019) study by Cerrillo-Briones and Ricardez-Sandoval [13] also implements rigorous UQ for the MEA system. Rather than analysis of an existing unit, this study is focused on optimal design of an absorber column under uncertainty in process variables and model parameters. This paper accounts for uncertainties in flue gas component flowrates ( $\mathrm{CO}_{2}$ and $\mathrm{N}_{2}$ ) and temperature, which are expected to be present due to variability in the operation of fossil fuel-based power plants; variability in the $\mathrm{CO}_{2}$ capture efficiency on the optimal plant design is also explored here. The treatment of model parametric uncertainty in this paper is limited to the individual activity coefficients of the species in the solvent ( $\mathrm{MEA}, \mathrm{H}_{2} \mathrm{O}, \mathrm{CO}_{2}$ ), which are each represented as single parameters with uniform uncertainty. This differs from our previous work [10], in which the activity coefficients are calculated as functions of the solvent properties, namely temperature and composition, and uncertainty is quantified directly for the parameters of the thermodynamic model using experimental observations.

Pilot plant testing is essential for demonstrating the capabilities of novel $\mathrm{CO}_{2}$ capture processes and validating models of such processes. Steady-state and dynamic data collection over large ranges of operating conditions is essential for ensuring robustness of process models. Experimental runs of chemical processes in pilot plants or at larger scales can require considerable financial resources and be time-consuming. Therefore, it is important to allocate resources optimally for maximizing the value of the information from these experimental campaigns. In an effort to validate the previous work for the CCSI MEA model, the submodels were combined into a full process model in Aspen Plus ${ }^{\text {® }}$, specifically for the Pilot Solvent Test Unit (PSTU) at the National Carbon Capture Center (NCCC) in Wilsonville, Alabama. This model was validated with data collected at NCCC in 2014 and shown to be largely predictive over a wide range of operating conditions [12]. Moreover, the steady-state model was used to develop a dynamic model, which was also validated with data collected at NCCC [14,15]. However, due to prior data limitations, the model did exhibit slight weakness in accurately capturing turndown of $\mathrm{CO}_{2}$ capture percentage. This data limitation was attributed to the choice of experimental design, in which a space filling approach was used to choose values for the controlled variables (flowrates of solvent, flue gas, and reboiler steam) for each test case without consideration of the output space. Furthermore, the stochastic modeling for the NCCC process indicated substantial uncertainty in some regions of its overall operating space.

As noted in Kimaev et al. [5], epistemic uncertainty can be reduced through further analysis of the system, in contrast to aleatory uncertainty, which is characterized by naturally random phenomena, and is thus irreducible. In the context of chemical process systems, the reduction of epistemic uncertainty can be actuated by collection of additional data at different scales. For example, if certain physical property models for a chemical process are considered to have high uncertainty, additional bench-scale experiments may be performed to gain more precise knowledge of those properties. On the other hand, other sources of uncertainty (e.g. mass transfer in a packed column) are dependent upon the process itself, including the specific design of the equipment items, and require data collected at pilot or industrial scale for rigorous quantification. With the incentive of reduction of process model uncertainty, the CCSI ${ }^{2}$ program planned and executed a new MEA test campaign at NCCC in the summer of 2017 with the goal of
using the existing CCSI ${ }^{2}$ process model to strategically collect data, which were then used to update the process model through Bayesian experimental design. This process, which will be referred to as sequential design of experiments (SDoE), involves selecting test points based on user-specified optimality criteria or utility functions, running the pilot plant according to the test plan, and using the resulting experimental data to update the distributions of certain submodel parameters. This experimental procedure is sequential because as the parametric distributions, through the process of Bayesian inference, are updated using experimental data, new estimates of the predicted uncertainty are obtained, leading to adjustment in the next experimental design. If the initial experimental design is updated sequentially by incorporating the knowledge learned during the early test runs, then the number of test runs can potentially be reduced and the overall quality of the final information be improved. In the SDoE process, the specific objectives of the test runs are decided by the researchers/ technology developers. For example, in this work, it was desired to reduce the uncertainty of the model's predictions, since reduction of model uncertainty is needed to improve confidence in the models as they are used for further analysis, including optimization and economic analysis. Therefore, the test runs were chosen to focus on operating conditions where the model has relatively high parametric uncertainty, which can be reduced through data collection. Due consideration of the uncertainties in the data and in the model must be made when incorporating the information from the test runs to update the process model, and when designing the new test plan with the updated process model.

In 1995, Bayesian experimental design was highlighted in a review paper by Chaloner and Verdinelli [16], which notes that much of the work in this field is more focused on theory than applications, and that very few examples of Bayesian design have been demonstrated in ongoing experiments. However, some applications have been demonstrated in recent years, including in chemical and biological engineering. For example, Bayesian experimental design is applied to the production of nitrile butadiene in a reactor system in Scott et al. [17] and to the measurement of combustion kinetics in Bisetti et al. [18]. Another application of this technique is given in Ryan et al. [19] for a pharmacokinetic modeling study. In a study by Atkinson and Bogacka [20], Bayesian designs are used for determining chemical reaction rates and orders. Chen et al. [21] have developed an optimization framework that uses existing process models for designing experiments, and demonstrated applications to a spray coating process and a reactor cascade system. The work of Kreutz and Timmer [22] provides a review of the use of model-based experimental design in the field of systems biology. Solonen et al. [23] presents an approach for Bayesian experimental design that allows for reduction in the number of experiments required to improve the precision of a stochastic model. Their approach is demonstrated for applications in kinetics, heat transfer, and reactor modeling using computationally simple, yet relevant, examples.

Few studies that apply Bayesian experimental design for CCS applications have been identified. One notable study is the work of Kalyanaraman et al. [24], which is based on a laboratory-scale process of $\mathrm{CO}_{2}$ adsorption on amine sorbents loaded in hollow fibers. In this study, uncertainty is quantified, through Bayesian inference, in the parametric space (six-dimensional) of the adsorption isotherm model and propagated through the system model, resulting in uncertainty estimations of the $\mathrm{CO}_{2}$ sorption capacity throughout the experimental design space, which is limited to two dimensions (temperature and pressure). This paper acknowledges the potential for Bayesian experimental design to be executed in a sequential nature, but only characterizes the effect of two data points on the reduction of the uncertainty in the model output. In a study from Konomi et al. [25], a Bayesian design process is used in the development of an adaptive sampling procedure for improving surrogate models. This work includes a case study on development of a surrogate for the output, namely the discretized distribution function of the solid volume
fraction, of a computational fluid dynamics (CFD) model of a regenerator column in a sorbent-based $\mathrm{CO}_{2}$ capture process. This study also uses a two-dimensional input space (sorbent particle diameter and gas velocity). The CCSI ${ }^{2}$ effort in application of SDoE to pilot-scale testing of $\mathrm{CO}_{2}$ capture processes was briefly summarized in the work of Soepyan et al. [26]. To the best of the authors' knowledge, no additional applications of Bayesian experimental design for pilot or plantscale chemical processes are available in the literature.

Due to the computational expense of many stages of the SDoE methodology, most notably the execution of a process simulation over a large input and parametric space and the Bayesian inference procedure used to refine parameter estimations, reduced-order modeling is necessary for making the process practical. A significant amount of recent work has focused on the use of machine learning (ML) and artificial intelligence (AI) applications in CCS. Li et al. [27] uses bootstrap aggregated neural networks for prediction of $\mathrm{CO}_{2}$ absorption as a function of key process inputs around the absorber column (flowrate, composition, temperature, and pressure of both flue gas and liquid streams). Sipöcz et al. [28] has developed artificial neural network (ANN) models of a complete solvent-based $\mathrm{CO}_{2}$ capture system that incorporates both multiple input and multiple output variables. Zhou et al. [29] presents work on development of neural networks for process data collected at the International Test Centre of $\mathrm{CO}_{2}$ Capture (ITC) in Canada. Hemmati et al. [30] use RSM techniques in the optimization of a rate-based absorber column model for the aqueous MEA solvent system. Much of the existing work with AI/ML in CO2 capture applications is primarily focused on prediction of physical properties rather than process performance, and a few of these are summarized here. Liu et al. [31] use ANN to predict $\mathrm{CO}_{2}$ solubility for several amine solvent systems. In separate works, Mesbah et al. [32] and Venkatraman and Alsberg [33] use machine learning methods to predict miscibility and solubility, respectively, of $\mathrm{CO}_{2}$ in various ionic liquids. The work of Yarveicy et al. [34] compares the results of various machine learning approaches to predicting $\mathrm{CO}_{2}$ equilibrium in the aqueous piperazine solvent system. $\mathrm{AI} / \mathrm{ML}$ has also been applied to work in $\mathrm{CO}_{2}$ sequestration; for example, Kim et al. [35] have used ANN to predict storage efficiency in deep saline aquifers. Our work, however, is focused mostly on existing surrogate modeling techniques that are available as part of the CCSI Toolset rather than rigorously comparing multiple AI/ML methodologies or developing novel techniques. It is also notable that there are no known applications of $\mathrm{AI} / \mathrm{ML}$ techniques in $\mathrm{CO}_{2}$ capture applications in which surrogates of a process are developed over an input space that includes both process operating variables and model parameters, which is a requirement for the SDoE procedure outlined in this work.

In addition to demonstrating SDoE for solvent-based $\mathrm{CO}_{2}$ capture systems, this work also seeks to contribute to the literature additional steady-state data, representing a wide range of process operating conditions, for the aqueous MEA system. The industrial standard solvent system for $\mathrm{CO}_{2}$ capture applications is generally considered to be an aqueous solution with $30 \mathrm{wt} \%$ MEA [36], and therefore a substantial amount of data for the operation of the process at various scales has been reported and many of these sources are summarized here. Due to the vast amount of literature in pilot testing for the aqueous MEA system, it is only possible to summarize some of the sources here. Some recommended sources for more exhaustive reviews of pilot plant studies include Cousins et al. [37] and Gelowitz et al. [38] Moreover, Bui et al. [39] has tabulated a list of existing studies for dynamic operation and modeling of post-combustion $\mathrm{CO}_{2}$ capture plants. In another work, Bui et al. [40] presents a comprehensive review of dynamic modeling and optimization for $\mathrm{CO}_{2}$ capture plants.

Data collected at Norway's Technology Centre Mongstad (TCM), which contains a 12 MWe test unit that is one of the world's largest facilities for testing carbon capture technologies, has been reported in various sources. In Brigman et al. [41], data are collected from TCM over a wide range of operating conditions, notably including variation in absorber packing height and some testing at a higher amine
concentration (40\%) than typically used, for a baseline of 85\% capture. Gjernes et al. [42] have provided a summary of twelve test series conducted at TCM, representing variation in absorber packing height, flowrates of solvent and flue gas, $\mathrm{CO}_{2}$ concentration in flue gas, and stripping temperature. The work of Faramarzi et al. [43] has established a baseline operation condition for the MEA process at TCM based on the minimization of solvent regeneration energy and presents the data for this baseline in great detail. This work is notable for its detailed analysis of uncertainty in $\mathrm{CO}_{2}$ capture efficiency, determined from the estimated measurement uncertainties of the $\mathrm{CO}_{2}$ flowrate in the absorber inlet, absorber outlet, and stripper outlet; this paper also compares four distinct measurements of the $\mathrm{CO}_{2}$ capture efficiency. In Montañés et al. [44], ten steady-state test cases of TCM data have been presented along with dynamic data for testing of the open-loop transient response of the plant and the performance of decentralized control structures. In a recent study, Bui et al. [45] presents three dynamic scenarios (effect of steam flowrate, time-varying solvent regeneration, and variable ramp rate) for testing with MEA at TCM along with a complete dynamic dataset.

In addition to the process data collected for the MEA system at the large pilot-scale TCM system, various test campaigns have been reported for small pilot-scale systems. Bui et al. [39] have presented data for two steady-state test campaigns along with three dynamic scenarios from the UK Carbon Capture and Storage Research Center (UKCCSRC) Pilot-scale Advanced Capture Technology (PACT) pilot plant (estimated scale of 60 kWe ). The work of Mangalapally and Hasse [46] has presented three sets of parametric studies performed on the MEA system at the pilot plant at the University of Kaiserslautern (estimated scale of 10 kWe), with variation included for the solvent and gas flowrates, $\mathrm{CO}_{2}$ removal rate, and $\mathrm{CO}_{2}$ partial pressure in flue gas. Data from two reference experiments with MEA at this same plant have been presented in Notz et al. [47] Sønderby et al. [48] have presented data for a pilot absorber column (estimated scale of 10 kWe ) with a total of 23 test runs representing variation in absorption height, solvent flowrate, and CO2 loading in the solvent inlet to the absorber. Dugas et al. [49] have presented work on validation of an absorber model at with twelve runs of pilot data at 1.2 MWe scale representing variation in solvent flowrate and $\mathrm{CO}_{2}$ loading. The work of Moser et al. [50] has described a 5000 h test campaign with MEA at the pilot plant (estimated scale of 0.35 MWe) at RWE Power's Coal Innovation Centre in Niederaussem, Germany, including parametric studies for variation of solvent flowrate, desorber pressure, and interstage cooling in the absorber. Zhang et al. [51] have data sets collected from a pilot plant at the University of Texas at Austin (estimated scale of 0.1 MWe ) representing 24 operating conditions including variation in solvent and gas flowrates, stripper pressure, lean $\mathrm{CO}_{2}$ loading, and column packing type. Artanto et al. [52] have presented data for MEA and amine-blend systems collected at the CSIRO pilot plant (estimated scale of 40 kWe ) at the Loy Yang Power station in Australia, including variation in solvent and flue gas flowrates and stripper bottom temperature. Tobiesen et al. have presented, in separate papers for absorber [53] and stripper [54] operation, 20 data points for MEA process operation at a scale of approximately 50 kWe representing a wide range of process conditions. Mejdell et al. [55] summarizes results of a MEA test campaign at the SINTEF Tiller pilot plant in Norway (estimated scale of 60 MWe ), which included 71 test runs with variation in flue gas flowrate and $\mathrm{CO}_{2}$ concentration (including both coal and natural gas exhaust conditions), liquid circulation rate, and reboiler duty. Koller et al. [56] presented test results of a CO2 spray scrubbing system with MEA (estimated scale of 0.05 MWe ) representing variation in liquid flowrate, $\mathrm{CO}_{2}$ capture rate, and reboiler duty. In a study on model development and process improvements for the aqueous MEA system, Li et al. [57] present detailed analysis of data collected at the pilot plant at Australia's Tarong power station (estimated scale of 0.1 MWe ).

Collectively, the extensive amount of literature on pilot plant testing for the MEA system includes data representing various scales and
process configurations as well as wide ranges of operating conditions. However, these sources vary widely both in terms of the number of data presented as well as the form in which they are presented (e.g. graphical and/or tabular). The supplementary data of Mangalapally and Hasse [46] includes very detailed process data for 19 operating points. The supplementary material of our previous work [12] presents tabulated data sets for 23 test runs executed at NCCC. Faramarzi et al. [43] and Notz et al. [47] both include very detailed data sets for baseline cases. The works of Bui et al. [39,45] include detailed dynamic as well as steady-state process data. Many other papers include tabulated data for the full process over a variety of operating conditions [42,44,57]. Some works include many data sets that are presented primarily in graphical form [41,50,52,55,56]. Others include experimental data focused specifically on the absorption [48,49,51,53] or stripping [54] section of the process. This paper aims to present a relatively large number of data collected over a wide variety of operating conditions tabulated in sufficient detail so that they are accessible to other researchers for use in model validation and comparison for the absorption and stripping sections of the process.

The available literature also includes many valuable insights into strategies for designing test campaigns with various types of objectives. However, the specific test objective of interest to the CCSI ${ }^{2}$ team, particularly targeting collection of data for reduction of process model parametric uncertainty, has not been previously demonstrated. The available literature for MEA system modeling is generally lacking in rigorous parametric uncertainty analysis, and the previous modeling efforts are generally deterministic in nature, meaning that process model inputs and outputs are represented as point values without including epistemic uncertainty in the model prediction. Although previous CCSI work [12] established a stochastic model of the MEA system validated with pilot plant data, this did not incorporate any methodology for reduction of the predicted uncertainty in the model outputs. Moreover, the test campaign described in the previous work was limited in that the experimental design considered only space filling in the process input (flowrates of solvent, flue gas, reboiler steam), resulting in clustering in $\mathrm{CO}_{2}$ capture percentage, an important output for the absorption section of the process. This paper seeks to complement our previous work by incorporating design of experiments to produce a new data set with less clustering in $\mathrm{CO}_{2}$ capture percentage, while introducing the SDoE methodology that combines strategic selection of data with reduction of model uncertainty. It is important to note that the applicability of the specific SDoE process described in this work is contingent upon the availability of a stochastic process model, and is not assumed to be applicable for all test campaign goals. Finally, more effort was made to include data collection for absorber operation with variable packing height, which was limited during the previous NCCC test campaign, to allow for further model validation while operating the column at lower packed height.

In summary, the most important contributions of this work include:
- Proposed a novel methodology for sequential design of experiments (SDoE) which allows for use of a stochastic process model to inform collection of plant data, which are in turn used to refine the process model in a cyclical manner.
- Demonstration of the SDoE process in a test campaign with aqueous MEA conducted at the 0.5 MWe pilot at NCCC, with the specific goal of using $\mathrm{CO}_{2}$ capture percentage data to refine parametric distributions, specifically those for mass transfer and hydraulics submodels.
- Presentation of 29 data sets for the aqueous MEA carbon capture process, representing a wide range of operating conditions including variation in solvent and flue gas flowrates, CO2 capture rate, CO2 concentration in flue gas, reboiler duty, and packing height and use of intercooling in the absorber column.

\section*{2. Methodology}

As described in our previous work [12], a steady-state test campaign for $\mathrm{CO}_{2}$ capture with aqueous MEA was executed at the National Carbon Capture Center (NCCC) in the summer of 2014. A total of 23 tests were performed, spanning a wide range of operating conditions, including changes to the flowrates of flue gas, circulated solvent, and steam input to the stripper reboiler. Moreover, the packing height of the absorber column in use was varied throughout the test campaign by changing the position at which the solvent enters the absorber, and solvent intercooling stages were used for some of the test cases. These data were used to validate the deterministic and stochastic models. One issue that was observed with those test runs was clustering in the CO2 capture percentage in the absorber column; capture percentage exceeded $95 \%$ for 16 of the 23 test runs and $99 \%$ for 8 of the runs. This clustering was attributed to the use of the space-filling design that considered only the manipulated process input variables (flowrates of solvent, flue gas, and reboiler steam) without the use of a model to incorporate the process outputs (e.g. $\mathrm{CO}_{2}$ capture percentage). Therefore, these runs did little to refine understanding outside this narrow range of operation. Furthermore, it was observed that the model prediction uncertainty was rather high at certain operating conditions due to the relatively small amount of data available to explore a higher dimensional input space. If the model is used for design and/or operation of the capture system, then high model uncertainties reflect higher technological risk and/or suboptimal operation. For the campaign described in this work, the existing process model is used to strategically plan the test conditions for which data are collected based on a sequential Bayesian design of experiments. The general procedure for SDoE is discussed in the following section.

\subsection*{2.1. General sequential design of experiments process}

The SDoE process is represented schematically in Fig. 1. Although the process is being applied to carbon capture systems in this work, the procedure is designed to be flexible for various process systems engineering applications for which stochastic models are used.

In this SDoE methodology, a stochastic model of the process of interest is required with some parameters, whose uncertainty is represented by probability density functions (PDFs). The model can be
denoted by:

$$
\begin{equation*}
y=f\left(\widetilde{x} ; \widetilde{\theta}_{1}, \widetilde{\theta}_{2}\right) \tag{1}
\end{equation*}
$$

where $f$ represents a full process model, $y$ the calculated output of interest, $\widetilde{x}$ the set of input variables, and $\widetilde{\theta}_{1}$ and $\widetilde{\theta}_{2}$ sets of model parameters of variable and static uncertainty, respectively. The parameter sets $\widetilde{\theta}_{1}$ and $\widetilde{\theta}_{2}$ differ in their treatment in the SDoE process; the distributions of the parameters contained in $\widetilde{\theta}_{2}$ remain fixed throughout the process whereas those of the parameters contained in $\widetilde{\theta}_{1}$ are updated in each iteration. The rationale of associating individual parameters to these groups is discussed in the forthcoming case study presented for the MEA campaign at NCCC. Due to the computationally expensive nature of process simulation models, the SDoE process generally requires that the full model be replaced by a response surface surrogate, represented as:

$$
\begin{equation*}
\hat{y}=\hat{f}\left(\widetilde{x} ; \widetilde{\theta}_{1}, \widetilde{\theta}_{2}\right) \tag{2}
\end{equation*}
$$


The surrogate model $\widehat{f}$ is developed to serve as a reduced order emulator of the rigorous model $f$. The development of this response surface model requires a large sample of simulation output representing the value of output variable $y$ calculated over the full space of interest of input variables and model parameters. This sample is used for training and validating the surrogate model, for which the predicted output value $\widehat{y}$ should closely approximate the simulation output across the whole input space of interest $\left\{\widetilde{x}, \widetilde{\theta}_{1}, \widetilde{\theta}_{2}\right\}$. A 10 -fold cross validation procedure is used for building the model. Many computational methods for developing surrogate models are available in the Problem Solving environment for Uncertainty Analysis and Design Exploration (PSUADE) [58], which is an independent software that has been included in the CCSI Toolset. Some examples of surrogate modeling techniques available in PSUADE include polynomial regression, Multivariate Adaptive Regression Splines, Gaussian process, radial basis function, k-nearest neighbors and sum of trees. In addition to the surrogate model, initial estimates, or prior distributions, of all model parameters are required to execute SDoE; these are denoted as $P\left(\widetilde{\theta}_{1}\right)$ and $P\left(\widetilde{\theta}_{2}\right)$ for the parameters of variable and static uncertainty. It is assumed that the parameters in these two sets are uncorrelated. Finally, a candidate set of points from across the input space of interest $\left(\widetilde{x} \ni \widetilde{x}^{(i)} ; \forall i=1, \cdots, N\right)$ should be identified for consideration for

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3caa1e19-e3fa-41ca-9661-b80d49821925-05.jpg?height=680&width=1593&top_left_y=1700&top_left_x=238}
\captionsetup{labelformat=empty}
\caption{Denotes input to SDoE algorithm}
\end{figure}

\begin{figure}
\captionsetup{labelformat=empty}
\caption{Denotes input to SDoE algorithm}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3caa1e19-e3fa-41ca-9661-b80d49821925-05.jpg?height=90&width=1206&top_left_y=2445&top_left_x=495}
\end{figure}

Fig. 1. Schematic of Bayesian sequential design of experiments methodology implemented for pilot plant campaign.
inclusion in pilot testing. The candidate set should be fully representative of the possible operating space of interest for the pilot testing campaign, taking into account any known constraints. For each $\tilde{x}^{(i)}$ in the candidate set, the stochastic model is evaluated in order to determine the predicted uncertainty for that point. This uncertainty is estimated by propagating influence of the full set of parameters $\left(\tilde{\theta}=\left[\begin{array}{ll}\tilde{\theta}_{1} & \tilde{\theta}_{2}\end{array}\right]\right)$ through the model. A sample of size $n_{1}$ is taken from the distributions of the full parameter space $\left(\tilde{\theta}^{(j)}, \forall j=1, \cdots, n_{1}\right)$ so that a set of output values $\left\{\hat{y}\left(\tilde{x}^{(i)} ; \tilde{\theta}^{(1)}\right), \cdots, \hat{y}\left(\tilde{x}^{(i)} ; \tilde{\theta}^{\left(n_{1}\right)}\right)\right\}$ is obtained by evaluating the surrogate model $\widehat{f}$. A confidence interval of level $100(1-\alpha) \%$ can be constructed from the set of output values:

$$
\begin{align*}
\left.C I^{\alpha}\right|_{\tilde{x}^{(i)} ; \tilde{\theta}_{1}, \tilde{\theta}_{2}}= & F_{1-\alpha / 2}\left(\left\{\hat{y}\left(\tilde{x}^{(i)} ; \tilde{\theta}^{(1)}\right), \cdots, \hat{y}\left(\tilde{x}^{(i)} ; \tilde{\theta}^{\left(n_{1}\right)}\right)\right\}\right) \\
& -F_{\alpha / 2}\left(\left\{\hat{y}\left(\tilde{x}^{(i)} ; \tilde{\theta}^{(1)}\right), \cdots, \hat{y}\left(\tilde{x}^{(i)} ; \tilde{\theta}^{\left(n_{1}\right)}\right)\right\}\right) \tag{3}
\end{align*}
$$


The $\ell^{\text {th }}$ percentile of the set of model evaluations is represented by $F_{\ell}$. In this work, the uncertainty in the model prediction at point $\tilde{x}^{(i)}$ is characterized with a $95 \%$ confidence interval $(\alpha=0.05)$. The confidence interval calculations are then used in a utility function to determine a subset ( $\widetilde{x}_{\text {test }} \subset \widetilde{x}$ ) of the candidate set for which to collect process data during the pilot plant testing. The sets $\widetilde{x}$ and $\widetilde{x}_{\text {test }}$ are of sizes $N$ and $M$, respectively, where $M \ll N$. The test set generally consists of only a small number of the candidate points due to the limited resources available for most pilot test campaigns. Furthermore, it is often desired to use a small test set for an SDoE iteration so as to allow for as many iterations as possible in the overall test campaign. The utility function refers to some criteria for the prioritization of data collection based on the experimental goals, and the different options will be discussed in more detail in the next section. As the experimental data ( $Z$ ) are collected through plant operation, Bayesian inference is performed as follows. For parameters for which distributions are not updated as data are collected, a sample of size $n_{2}\left(\tilde{\theta}_{2}^{(k)}, \forall k=1, \cdots, n_{2}\right)$ is drawn from their constant distribution $P\left(\widetilde{\theta}_{2}\right)$. For each sample point $\tilde{\theta}_{2}^{(k)}$, a posterior distribution for the remaining parameters ( $\tilde{\theta}_{1}$ ) is calculated:

$$
\begin{equation*}
\pi_{k}\left(\tilde{\theta}_{1} \mid Z, \tilde{\theta}_{2}^{(k)}\right) \propto P\left(\tilde{\theta}_{1}\right) L\left(Z \mid \tilde{\theta}_{2}^{(k)}, \tilde{\theta}_{1}\right) \tag{4}
\end{equation*}
$$


In Eq. (4), $P\left(\widetilde{\theta}_{1}\right)$ is the prior distribution of the parameters to be updated and $\pi_{k}\left(\tilde{\theta}_{1} \mid Z, \tilde{\theta}_{2}^{(k)}\right)$ is the posterior distribution of $\tilde{\theta}_{1}$ conditional on the observed experimental data and the value of $\widetilde{\theta}_{2}$ for sample $k$. This posterior distribution is obtained, typically via some variants of the Markov chain Monte Carlo (MCMC) method [59], in the form of a set of sample points. $L\left(Z \mid \tilde{\theta}_{2}^{(k)}, \tilde{\theta}_{1}\right)$ represents the likelihood (some distance metric to express the deviation between simulation and data) of observing a set of experimental data conditioned on the values of the parameters. The likelihood function used by the PSUADE software is [60]:

$$
\begin{equation*}
L\left(Z \mid \tilde{\theta}_{2}^{(k)}, \tilde{\theta}_{1}\right)=\exp \left(-0.5 \sum_{i=0}^{n_{\text {data }}} \frac{\left[\hat{y}\left(\tilde{x}_{i} ; \tilde{\theta}_{2}^{(k)}, \tilde{\theta}_{1}\right)-Z\left(\tilde{x}_{i}\right)\right]^{2}}{n_{\text {data }} \sigma_{i}^{2}}\right) \tag{5}
\end{equation*}
$$


In Eq. (5), which is the likelihood function based on the chi-square statistic, $Z\left(\widetilde{x_{i}}\right)$ represents the output variable value of the $\mathrm{i}^{\text {th }}$ data point, observed at the input condition $\tilde{x}_{i}$, and $\hat{y}\left(\tilde{x}_{i} ; \tilde{\theta}_{2}^{(k)}, \tilde{\theta}_{1}\right)$ represents the surrogate model evaluated at the same input condition with fixed values of the model parameters $\left(\tilde{\theta}_{1}\right.$ and $\left.\tilde{\theta}_{2}^{(k)}\right)$; $n_{\text {data }}$ is the total number of experimental data. The variance of the output variable value is given by $\sigma_{i}{ }^{2}$, and the presence of this term in the likelihood function allows for incorporation of process variable measurement uncertainty into the algorithm for updating the parametric uncertainty. Having obtained the $n_{2}$ individual sets of sample points that represent the individual $\pi_{k}$, the overall posterior distribution ( $\pi\left(\widetilde{\theta}_{1} \mid Z, \widetilde{\theta}_{2}\right)$ ) is obtained by combining all of the individual sets, a process called marginalization in statistics (Note: inference and marginalization can be combined into a single
process.) The posterior distribution is then used as the updated prior distribution for parameter set $\widetilde{\theta}_{1}$ in the next iteration of the SDoE procedure. At this point in the procedure, the user has the option of retaining the existing surrogate model or attempting to improve its accuracy in a refined region of interest. If the experimental goals change such that the user is no longer interested in collecting more data in a subset of input space $\tilde{x}$, a new surrogate model may be developed from a new sample of simulation results that include values of $\tilde{x}$ only in the refined region of interest. This could result in increased accuracy of the surrogate throughout the new input region of interest. Moreover, if some portion of the space of $\tilde{\theta}_{1}$ has been eliminated from the posterior distribution in a previous SDoE iteration, it may also be removed from the sample used to build the surrogate model in an effort to approve the surrogate's predictability in the refined region of interest for $\widetilde{\theta}_{1}$. Due to the time requirement for running a large number of simulations and using the results to develop surrogate models, it may be impractical to build new surrogate models between iterations of SDoE. Hence, it is very important to have available before the test campaign an adequate surrogate model that is accurate throughout the full space of input variable and model parameter values. Through use of the surrogate model, the $95 \%$ confidence intervals are re-evaluated with the additional information, generally resulting in a reduction of the estimated uncertainty throughout the input space. The updates of the estimated confidence intervals for the points in the candidate set are then used to choose the next set of test runs for which data are to be collected, thus resulting in the sequential nature of the process. The termination criteria for the SDoE algorithm may be chosen by the experimenter, and one practical choice would be to stop testing when the predicted output uncertainty throughout the design space no longer appreciably diminishes. For large-scale test campaigns, in practice, the number of iterations of SDoE may be limited by the overall time available for the campaign, as will be demonstrated in the results of this work.

\subsection*{2.2. Choice of utility function}

The SDoE process is flexible enough to accommodate many different types of utility functions or user-specified optimality criteria. The criteria on which to focus can be fixed throughout the SDoE process, or they can evolve in different stages. At each stage, one or more criterion can be considered, depending on the priorities of the stage.

There are many possible goals of a particular SDoE stage. Common choices are (a) exploration of response values throughout the input space, (b) refining the constraints on the input region of interest, (c) improving the precision (or reducing the uncertainty) in the estimation of model parameters, (d) improving the precision of prediction for new observations in the design region, (e) quantifying the discrepancy between the model and data, or (f) optimizing the value of responses of interest. The computational complexity required for implementation into SDoE varies significantly for the different types of goals. Particularly, the goals that are focused on model input and output uncertainty ( $c$ and d) require use of stochastic models, which are generally more cumbersome to develop and implement than their deterministic counterparts.

The choices above are listed in an order that might be typical of many SDoE plans. Initially, there may be interest in understanding some of the fundamentals of the region of interest and what response values are possible (a or b). Space-filling designs [61] are common for this initial exploration with either a minimax or maximin criterion [62] being used as the utility function. Then interest shifts to refining the model, where interest may focus either on estimation or prediction (c or d). Here the goal is to increase confidence in the estimated response surface with minimal uncertainty on the individual model parameters, often using D- and A-optimality [63], which are dependent on the model form and complexity selected. Improving the precision of prediction for new observations, involves criteria that evaluate the
prediction variance throughout the input region of interest, which occurs by examining the average (I-optimality) or maximum (G-optimality) prediction variance in the region [63]. For example, if the uncertainty ranges for an output variable are known for different points within an input space and the extent of the uncertainty varies considerably among the points, the concept of G-optimality can be applied to the experimental design so that points for which the predicted uncertainty is high can be targeted for selection. Stated otherwise, G-optimality seeks to minimize the maximum predicted uncertainty in the output variable over the full input space through strategic selection of points for which the model predicts relatively high uncertainty.

In early stages of data collection, if there are differences between the model and the observed data, then trying to reduce the size of that discrepancy by obtaining more data in regions of high discrepancy might be warranted (e). Finally, once the performance of the model is satisfactory for its intended use, a final stage often involves collecting more data close to where the process might operate under ideal conditions to get the best possible response values.

For the MEA campaign at NCCC, initial stages of the experiment focused on exploration throughout the input space of interest, and later stages had the goals of maximally improving the quality of prediction of observations throughout the region, using the aforementioned concept of G-optimality.

\subsection*{2.3. Modeling of National Carbon Capture Center pilot plant}

The National Carbon Capture Center (NCCC), located in Wilsonville, Alabama, USA, is a test facility with the capability of evaluation of a wide variety of processes at large ranges of equipment sizes and process conditions; coal-derived flue gas from the adjoining Alabama Power Gaston Plant is available at the facility [64]. A schematic of the 0.5 MWe pilot solvent test unit (PSTU) post-combustion carbon capture process, which has a $\mathrm{CO}_{2}$ capture capacity of 10 ton/day, is shown in Fig. 2. This schematic includes only the major equipment in the absorption and stripping section, which is of primary interest for this work.

The flue gas supplied from the power plant is passed through a pre-
scrubber for removal of $\mathrm{SO}_{2}$ and a cooler/condenser for reduction of the $\mathrm{H}_{2} \mathrm{O}$ content and decrease in temperature. The flue gas enters the bottom of the absorber column and is contacted countercurrently in the packing material by the solvent flowing down the column. The $\mathrm{CO}_{2^{-}}$ lean solvent enters the column through one of three inlets as shown in the schematic; due to the presence of multiple inlets, the total height of packing for gas/liquid contact is variable in the PSTU operation. CO2 is removed from the flue gas by reactive absorption. Due to the exothermic nature of the reaction between $\mathrm{CO}_{2}$ and the amine solvent, the temperature in the column increases and reduces the driving force for absorption of the $\mathrm{CO}_{2}$. As the absorber packing height, and thus the amount of area available for mass transfer, increases, the $\mathrm{CO}_{2}$ capture efficiency also increases until further uptake of $\mathrm{CO}_{2}$ into the liquid is impeded by a pinch point in the column. At the pinch point, the equilibrium $\mathrm{CO}_{2}$ pressure of the liquid phase approaches the $\mathrm{CO}_{2}$ partial pressure of the vapor phase, and the driving force for absorption approaches zero. Intercooling stages, which are optionally used during process operation, are located between the beds. The purpose of these stages is to remove a portion of the solvent exiting a bed, cool it to a temperature typical for absorption, and return it to the next bed further down the column. Intercooling results in the general effect of increasing the driving force for absorption in the middle of the column, and thus increasing the $\mathrm{CO}_{2}$ capture efficiency of the absorber. The $\mathrm{CO}_{2}$-rich solvent exits the bottom of the column and the clean flue gas exits at the top, and is sent to a washing tower for reduction of water content before being emitted to the atmosphere.

The $\mathrm{CO}_{2}$-rich solvent exiting the absorber is heated in the lean/rich heat exchanger by the hot lean solvent exiting the stripper, to a temperature favorable for stripping the absorbed $\mathrm{CO}_{2}$ from the solvent. In the stripper, $\mathrm{CO}_{2}$ is removed from solvent via an endothermic reaction, for which the energy is provided by steam input to the reboiler. The vapor stream exits the top of the stripper and is cooled to about $40^{\circ} \mathrm{C}$ in the condenser, resulting in separation of $\mathrm{CO}_{2}$ and $\mathrm{H}_{2} \mathrm{O}$. The vapor phase, primarily $\mathrm{CO}_{2}$, is sent for compression and sequestration and the liquid, primarily $\mathrm{H}_{2} \mathrm{O}$, is sent back to the stripper as reflux. The $\mathrm{CO}_{2^{-}}$ lean solvent exits the bottom of the stripper and is partially vaporized in the reboiler, with the boilup sent back to the stripper, providing the

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3caa1e19-e3fa-41ca-9661-b80d49821925-07.jpg?height=929&width=1683&top_left_y=1606&top_left_x=193}
\captionsetup{labelformat=empty}
\caption{Fig. 2. Schematic of absorption/stripping section of the Pilot Solvent Test Unit at National Carbon Capture Center.}
\end{figure}

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 1
Equipment specifications for NCCC PSTU.}
\begin{tabular}{|l|l|}
\hline Specification & Value \\
\hline Absorber Packing Height (m) & 18 ( 3 beds of 6 m ) \\
\hline Absorber Diameter (m) & 0.64 \\
\hline Stripper Packing Height (m) & 12 (2 beds of 6 m ) \\
\hline Stripper Diameter (m) & 0.59 \\
\hline Absorber/Stripper Packing Type & MellapakPlus 252Y \\
\hline Lean/Rich Heat Exchanger Area ( $\mathrm{m}^{2}$ ) & 114 \\
\hline
\end{tabular}
\end{table}

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 2
Ranges of operating conditions for NCCC PSTU during 2014 MEA test campaign.}
\begin{tabular}{|l|l|}
\hline Variable & Values/Range \\
\hline Flue Gas Flowrate (kg/h) & 1300-3000 \\
\hline Flue Gas Temperature ( ${ }^{\circ} \mathrm{C}$ ) & 41-48 \\
\hline Flue Gas $\mathrm{CO}_{2}$ Concentration (mol\% - dry) & 8.2-11.6 \\
\hline Lean Solvent Flowrate (kg/h) & 3100-12000 \\
\hline Lean Solvent Temperature ( ${ }^{\circ} \mathrm{C}$ ) & 40-50 \\
\hline Lean Solvent $\mathrm{CO}_{2}$ loading ( $\mathrm{mol} \mathrm{CO}_{2} / \mathrm{mol}$ MEA) & 0.06-0.40 \\
\hline Lean Solvent MEA Concentration (\%) & 27-33 \\
\hline L/G Ratio (by mass) & 1.4-8.2 \\
\hline Absorber packing height (meter) & 6/12/18 \\
\hline Intercooling stages & ON/OFF \\
\hline $\mathrm{CO}_{2}$ Capture Rate (\%) & 54-99.9 \\
\hline Stripper Pressure (kPa) & 179-185 \\
\hline Reboiler Duty (kW) & 160-680 \\
\hline
\end{tabular}
\end{table}
heat input for the stripping, and the lean solvent is sent to the lean/rich heat exchanger, where it is cooled by the cold rich solvent exiting the absorber column. The solvent passes through an additional cooler to reduce the temperature to a typical absorption condition, and is then supplied to the feed tank, from which the solvent is supplied to the absorber. Table 1 includes equipment specifications that are necessary for steady-state modeling of the system.

A steady-state model of this process was developed as described in our previous works [9-12], using Aspen Plus software along with the aforementioned CCSI Toolset to enable stochastic modeling capabilities through UQ. UQ was first applied to stand-alone property models for the MEA- $\mathrm{CO}_{2}-\mathrm{H}_{2} \mathrm{O}$ system, including viscosity, density, and surface tension [9]. The thermodynamic framework was developed using the built-in electrolyte Non-Random Two-Liquid (e-NRTL) method, and vapor-liquid equilibrium, heat capacity, and heat of absorption data were simultaneously regressed to fit the model and perform UQ [10]. A novel simultaneous regression approach was used to quantify the uncertainty for mass transfer, diffusivity, interfacial area, and reaction kinetics using both wetted-wall column and packed column data [11]. Finally, stochastic models for the packing hydraulics were also developed [11]. All of these submodels were integrated into a full process model for the PSTU at NCCC, which was validated using data collected during a steady-state test campaign held at NCCC in 2014 [12]. This previous work included separate validation of the absorber and stripper models by comparing model predictions with data for percentage of $\mathrm{CO}_{2}$ capture by the absorber and lean solvent $\mathrm{CO}_{2}$ loading in the stripper outlet; the percentage error was found to be within $\pm 5 \%$ and $\pm 10 \%$, respectively, for these quantities in most cases. Furthermore, the simulated temperature profiles of the absorber and stripper
![](https://cdn.mathpix.com/cropped/3caa1e19-e3fa-41ca-9661-b80d49821925-08.jpg?height=573&width=696&top_left_y=1281&top_left_x=1070)

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3caa1e19-e3fa-41ca-9661-b80d49821925-08.jpg?height=1205&width=1413&top_left_y=1278&top_left_x=288}
\captionsetup{labelformat=empty}
\caption{Fig. 3. Model calculation of $\mathrm{CO}_{2}$ capture percentage in absorber for variable liquid and gas flowrates and lean loading.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3caa1e19-e3fa-41ca-9661-b80d49821925-09.jpg?height=1360&width=1683&top_left_y=180&top_left_x=189}
\captionsetup{labelformat=empty}
\caption{Fig. 4. Stochastic model estimation of widths of $95 \%$ confidence intervals of $\mathrm{CO}_{2}$ capture prediction for variable liquid and gas flowrates and lean loading.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3caa1e19-e3fa-41ca-9661-b80d49821925-09.jpg?height=616&width=792&top_left_y=1632&top_left_x=167}
\captionsetup{labelformat=empty}
\caption{Fig. 5. Parity plot for comparison of $\mathrm{CO}_{2}$ capture percentage predicted by original simulation model and surrogate response surface model.}
\end{figure}
columns, which varied widely for different column operating conditions, were compared to test data and shown to adequately capture the trends for most cases.

The complete data sets and comparison of the absorber and stripper temperature profiles for 23 cases and can be found in the supporting

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 3
First set of cases selected for 2017 MEA test campaign at NCCC.}
\begin{tabular}{|l|l|l|l|l|l|l|}
\hline Case No. & Lean Solvent Flowrate (kg/h) & Flue Gas Flowrate (kg/h) & Lean Solvent Loading (mol $\mathrm{CO}_{2} / \mathrm{mol}$ MEA) & Reboiler Duty (kW) & Flue Gas $\mathrm{CO}_{2}$ Mole Fraction & $\mathrm{CO}_{2}$ Capture Percentage (Model Prediction) \\
\hline 1* & 3911 & 1250 & 0.30 & 151 & 0.117 & 77.3 \\
\hline 2 & 3200 & 2250 & 0.25 & 157 & 0.093 & 54.8 \\
\hline 3* & 3800 & 2500 & 0.15 & 264 & 0.105 & 72.9 \\
\hline 4* & 9384 & 3000 & 0.25 & 470 & 0.117 & 89.3 \\
\hline 5 & 4171 & 3000 & 0.10 & 465 & 0.117 & 69.6 \\
\hline 6* & 6817 & 2250 & 0.30 & 264 & 0.117 & 72.8 \\
\hline 7 & 8186 & 3000 & 0.25 & 414 & 0.082 & 96.1 \\
\hline 8 & 3133 & 1750 & 0.30 & 120 & 0.082 & 61.0 \\
\hline 9* & 7946 & 3000 & 0.20 & 486 & 0.105 & 97.3 \\
\hline 10 & 3017 & 2750 & 0.10 & 336 & 0.105 & 60.8 \\
\hline 11 & 6514 & 2500 & 0.25 & 323 & 0.117 & 78.6 \\
\hline 12 & 3609 & 3000 & 0.15 & 252 & 0.082 & 71.8 \\
\hline 13 & 8024 & 2500 & 0.25 & 406 & 0.105 & 96.3 \\
\hline 14* & 9384 & 3000 & 0.25 & 470 & 0.117 & 89.3 \\
\hline 15 & 3230 & 2250 & 0.10 & 360 & 0.117 & 72.3 \\
\hline 16 & 6932 & 2750 & 0.20 & 417 & 0.117 & 90.2 \\
\hline 17* & 4341 & 2000 & 0.20 & 259 & 0.105 & 87.7 \\
\hline 18 & 3360 & 1500 & 0.20 & 199 & 0.117 & 83.7 \\
\hline 19 & 3370 & 2750 & 0.15 & 234 & 0.117 & 53.9 \\
\hline 20 & 4734 & 2250 & 0.15 & 331 & 0.117 & 90.6 \\
\hline
\end{tabular}
\end{table}

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 4
Data obtained during first iteration of SDoE.}
\begin{tabular}{|l|l|l|l|l|l|l|l|}
\hline \multicolumn{4}{|c|}{} & Lean Solvent Loading (mol $\mathrm{CO}_{2} / \mathrm{mol}$ MEA) & Flue Gas $\mathrm{CO}_{2}$ Mole Fraction & \multicolumn{2}{|c|}{$\mathrm{CO}_{2}$ Capture Percentage} \\
\hline 1A & 7 & 8180 & 3000 & 0.24 & 0.082 & 97.5 & 97.2 \\
\hline 2A & 16 & 7130 & 2690 & 0.25 & 0.100 & 93.4 & 90.2 \\
\hline 3A & 18 & 3354 & 1500 & 0.24 & 0.108 & 79.7 & 77.0 \\
\hline 4A & 12 & 3600 & 3000 & 0.20 & 0.076 & 70.6 & 66.6 \\
\hline 5A & 19 & 3380 & 2750 & 0.20 & 0.107 & 53.8 & 50.2 \\
\hline 6A & 8 & 3130 & 1750 & 0.31 & 0.076 & 51.7 & 60.6 \\
\hline 7A & 20 & 4730 & 2255 & 0.23 & 0.109 & 72.5 & 73.0 \\
\hline 8A & 2 & 3230 & 2240 & 0.24 & 0.107 & 56.3 & 51.8 \\
\hline 9A & 15 & 3224 & 2245 & 0.14 & 0.108 & 74.2 & 72.9 \\
\hline 10A & 13 & 7980 & 2492 & 0.32 & 0.109 & 79.9 & 74.2 \\
\hline 11A & 10 & 3016 & 2761 & 0.16 & 0.096 & 60.5 & 55.7 \\
\hline 12A & 5 & 4170 & 2920 & 0.14 & 0.107 & 76.0 & 72.5 \\
\hline 13A & 16* & 6910 & 2680 & 0.26 & 0.108 & 80.6 & 80.9 \\
\hline 14A & 11 & 6505 & 2500 & 0.31 & 0.108 & 57.8 & 63.1 \\
\hline 15A & 13* & 8000 & 2494 & 0.32 & 0.108 & 76.8 & 74.6 \\
\hline
\end{tabular}
\end{table}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3caa1e19-e3fa-41ca-9661-b80d49821925-10.jpg?height=610&width=793&top_left_y=1033&top_left_x=174}
\captionsetup{labelformat=empty}
\caption{Fig. 6. Parity plot for comparison of model prediction of $\mathrm{CO}_{2}$ capture percentage to experimental data obtained during first iteration of SDoE.}
\end{figure}
information of the Morgan et al. [12] paper; the main manuscript contains summaries of all data cases and temperature profile comparisons for select cases. The ranges or discrete values of the major operating variables from the 2014 test campaign are given in Table 2.

Table 2 shows that a wide range of process conditions were used when conducting the 2014 test campaign at NCCC. A total of 23 data were collected during this test campaign, for which more details are available in the previous work [12].

\subsection*{2.4. Candidate set specification}

Four variables are considered for planning the test runs in the 2017 NCCC campaign: the lean solvent flowrate ( $L$ ), the flue gas flowrate ( $G$ ), the $\mathrm{CO}_{2}$ loading in the lean solvent $\left(\alpha_{\text {lean }}\right)$, and the mole fraction of $\mathrm{CO}_{2}$ (dry basis) in the flue gas ( $y_{\mathrm{CO}_{2}}$ ). These are chosen since they are the major variables that affect the absorber performance and can be varied either directly or indirectly when operating the plant. The choice of using the absorption section of the process as the basis for the experimental design is based on the use of mass transfer and hydraulics model parameters in the set of parameters for which uncertainty is updated (represented in Fig. 1 as $\widetilde{\theta}_{1}$ ). These submodels primarily affect the absorber column performance, whereas the thermodynamic model
parameters, which are known to have significant effect on the performance of both the absorber and the stripper, are included in the set of parameters for which uncertainty is fixed (represented in Fig. 1 as $\widetilde{\theta}_{2}$ ). More discussion on the specific parameters included in the analysis is forthcoming in the results section of this work. The input variable set specified for each run of the experiment is defined as:

$$
\tilde{x}=\left[\begin{array}{llll}
L & G & \alpha_{\text {lean }} & y_{\mathrm{CO}_{2}} \tag{6}
\end{array}\right]
$$


The ranges for the values of each of the variables are:

$$
\begin{equation*}
L \in[3000-13,000] \mathrm{kg} / \mathrm{h} \tag{7a}
\end{equation*}
$$


$$
\begin{equation*}
G \in[1000-3000] \mathrm{kg} / \mathrm{h} \tag{7b}
\end{equation*}
$$


$$
\begin{equation*}
\alpha_{\text {lean }} \in[0.10-0.30] \mathrm{mol} \mathrm{CO}_{2} / \mathrm{mol} \text { MEA } \tag{7c}
\end{equation*}
$$


$$
\begin{equation*}
y_{\mathrm{CO}_{2}} \in[0.082-0.117] \mathrm{mol} \mathrm{CO}_{2} / \mathrm{mol} \mathrm{FG} \tag{7d}
\end{equation*}
$$


These ranges were established based on the overall ranges for which data were collected in the 2014 test campaign (Table 2). A few of the previous data contain lean loading either above or below the limits given here, although $0.3 \mathrm{~mol} \mathrm{CO}_{2} / \mathrm{mol}$ MEA has been determined to be a reasonable choice for the maximum value due to the high inefficiency of operating the absorber column at higher values of lean loading; this will be demonstrated in Section 3 of this paper. On the other hand, operation with lean loading at or below $0.1 \mathrm{~mol} \mathrm{CO}_{2} / \mathrm{mol}$ MEA results in a relatively high reboiler duty requirement in the stripper column, and thus a high cost of operation. Since many of the test runs in the 2014 campaign were clustered with percentage of capture higher than 99\%, a desired range of 50-95\% capture was selected for the new test campaign to prevent such clustering. When developing a candidate set of points for testing, the test space was constrained to include only points for which the estimated capture percentage falls within these limits. For discrete values of the subset $\left(\widetilde{u}=\left[\begin{array}{lll}G & \alpha_{\text {lean }} & y_{\mathrm{CO}_{2}}\end{array}\right]\right)$ of the input variables, the values of the liquid flowrate for which the $\mathrm{CO}_{2}$ capture percentage is at its lower and upper limits of 50 and 95\%, respectively, were calculated from the deterministic Aspen Plus model. This is denoted as:

$$
\begin{equation*}
\left\{L_{\min }^{(i)}, L_{\max }^{(i)}\right\}=f\left(\widetilde{u}^{(i)}\right) ; \quad \forall i=1, \cdots n_{3} \tag{8}
\end{equation*}
$$


The full model was evaluated $n_{3}=36$ times, representing all combinations of three selected discrete values each of $G$ and $y_{\mathrm{CO}_{2}}$ and four discrete values of $\alpha_{\text {lean }}$; the additional discretization for the lean loading was used due to the highly sensitive and nonlinear relationship between this variable and the $\mathrm{CO}_{2}$ capture percentage. For any given $\widetilde{u}^{(i)}$ for which all elements lie within the ranges given in Eq. (7), the upper and lower limits of the liquid flowrate may be estimated by a trilinear interpolation function. This function, denoted by $\bar{f}$, is developed by using the $n_{3}$ model evaluations collected for discrete values of the input variables. This methodology is used to select a candidate set of test points using the following accept-reject algorithm. A point in the input space ( $\tilde{x}^{*}$ ) is chosen, with the individual variable values ( $x^{*}$ ) represented by:

$$
\begin{equation*}
x^{*} \sim U\left(x^{\min }, x^{\max }\right) ; \quad \forall x \in \tilde{x} \tag{9}
\end{equation*}
$$


Eq. (9) indicates that for each $x^{*}$ in the input set $\tilde{x}$, a value is independently sampled from a uniform distribution with upper and lower bounds of the variable, which are defined in Eq. (7). The point is accepted into the candidate test set if and only if the following condition is met:

$$
\begin{equation*}
\bar{L}_{\min } \leq L^{*} \leq \bar{L}_{\max } \tag{10}
\end{equation*}
$$


$$
\begin{equation*}
\left\{\bar{L}_{\min }, \bar{L}_{\max }\right\}=\bar{f}\left(\widetilde{u}^{*}\right) \tag{11}
\end{equation*}
$$

where $\widetilde{u}^{*}$ is the subset of variables in the candidate point $\widetilde{x}^{*}$ that includes all variables except the lean solvent flowrate and $L^{*}$ represents the value of this variable in the candidate point. Essentially, the value of the lean solvent flowrate must fall within the range predicted by the

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3caa1e19-e3fa-41ca-9661-b80d49821925-11.jpg?height=1045&width=1495&top_left_y=189&top_left_x=284}
\captionsetup{labelformat=empty}
\caption{Fig. 7. Estimated marginal PDFs for prior and posterior distributions of mass transfer and hydraulics model parameters updated through incorporating first set of test run data into Bayesian inference.}
\end{figure}
trilinear interpolation model for which the $\mathrm{CO}_{2}$ capture percentage is between 50 and $95 \%$ in order for the point to be included in the candidate set. This algorithm terminates when the candidate set has the specified number of test points.

\section*{3. Results}

\subsection*{3.1. Sensitivity studies and uncertainty quantification of the initial model}

The parameter distributions for thermodynamic, mass transfer, and hydraulics parameters were propagated through the absorber column model (with 3 beds and intercooling) for various points in the input space described by Eq. (7). For each point in the input space, the width of the $95 \%$ confidence interval was calculated. The results are shown for the deterministic model values of $\mathrm{CO}_{2}$ capture percentage in Fig. 3, and the confidence interval widths are given in Fig. 4. For the purpose of brevity, the results are shown only for flue gas with $10 \mathrm{~mol} \% \mathrm{CO}_{2}$. The values are shown for three values of flue gas flowrate, including the baseline value from the 2014 campaign ( $2250 \mathrm{~kg} / \mathrm{h}$ ) and the upper and lower limits of 1000 and $3000 \mathrm{~kg} / \mathrm{h}$, respectively.

The pronounced decline in absorber $\mathrm{CO}_{2}$ capture percentage with very high loading ( $\sim 0.4 \mathrm{~mol} \mathrm{CO}_{2} / \mathrm{mol} \mathrm{MEA}$ ) in the inlet solvent is shown in Fig. 4, hence the previously discussed choice of $0.3 \mathrm{~mol} \mathrm{CO}_{2} /$ mol MEA as an upper limit for the test plan.

\subsection*{3.2. Surrogate model development}

A surrogate model for the $\mathrm{CO}_{2}$ capture percentage, the variable to be used as the output variable for updating the parametric distributions through Bayesian inference, was developed using Multivariate Adaptive Regression Splines (MARS) [65], a non-parametric regression technique in which an output variable is represented as a summation of hinge functions to characterize the effect of individual input variables and products of hinge functions for the interaction effects of groups of input
variables. The development of an accurate surrogate model is essential for reducing the computational expense of the Bayesian inference procedure. This model was trained and validated with a total of 5773 Aspen Plus simulations of the process, across the entire input space of process variables and model parameters; a sample of 6000 was originally generated, and all sample points that resulted in failed simulations were not included when developing the surrogate model. A parity plot for the 10 -fold cross validation of the surrogate model is shown in Fig. 5. Based on the fit shown in Fig. 5, it was judged that the response surface model reasonably emulates the actual process model over the input space of interest; the correlation coefficient between the two models is $R^{2} \approx 0.995$. Therefore, it is considered an acceptable substitute for the actual Aspen Plus model for use in the computationally expensive SDoE procedure.

\subsection*{3.3. First iteration of sequential design of experiments}

The first test plan generated using the SDoE approach had a focus on exploration throughout the input region of operability, with design run details given in Table 3. It should be noted that while the lean solvent loading is treated as an input variable for the absorber model, it cannot be directly manipulated in the actual process. Therefore, the steam flowrate for the stripper reboiler required for reducing the solvent CO2 loading to the given value is estimated for each case. The estimated steam flowrate, along with the flowrates of the lean solvent and flue gas as well as the $\mathrm{CO}_{2}$ concentration in the flue gas, was used as the input for plant operation. A minimax criterion (minimizing the maximum distance between any of the candidate set locations and a selected design point) was used to select the 20 runs. This ensured that no region of the input space was ignored during the initial set of runs in this first stage.

The corresponding data collected from this test plan are summarized in Table 4, and more detailed plant data are given in Appendix A. The experimental data for $\mathrm{CO}_{2}$ capture percentage are compared with

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3caa1e19-e3fa-41ca-9661-b80d49821925-12.jpg?height=1217&width=1501&top_left_y=183&top_left_x=284}
\captionsetup{labelformat=empty}
\caption{Fig. 8. Effect of update in stochastic model using the first set of test run data on model prediction uncertainty ( $95 \%$ confidence interval widths for $\mathrm{CO}_{2}$ capture) for (A) the entire input space and (B) points for which data were collected.}
\end{figure}

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 5
Data obtained during second iteration of SDoE.}
\begin{tabular}{|l|l|l|l|l|l|l|}
\hline \multirow[t]{2}{*}{Data No.} & \multirow[t]{2}{*}{Lean Solvent Flowrate (kg/h)} & \multirow[t]{2}{*}{Flue Gas Flowrate (kg/h)} & \multirow[t]{2}{*}{Lean Solvent Loading (mol CO2/mol MEA)} & \multirow[t]{2}{*}{Flue Gas CO2 Mole Fraction} & \multicolumn{2}{|l|}{$\mathrm{CO}_{2}$ Capture Percentage} \\
\hline & & & & & Data & Model \\
\hline 1B & 7959 & 2497 & 0.30 & 0.077 & 96.1 & 93.5 \\
\hline 2B & 9871 & 2746 & 0.30 & 0.088 & 97.7 & 93.1 \\
\hline 3B & 11,412 & 2748 & 0.30 & 0.108 & 94.9 & 92.5 \\
\hline
\end{tabular}
\end{table}
model predictions, and the parity plot is given in Fig. 6.
A total of 15 experimental data sets were collected in the first round of this work. Data for some of the planned runs in Table 3 were not obtained due to issues with plant operation; these are denoted by * in the table. The order in which the data were obtained was modified for ease of process operation, and the data sets are presented in Table 4 in the order in which they were collected; the case numbers presented in Table 4 are used to match the data with the corresponding planned experiment given in Table 3. Two of the runs near the end of the SDoE iteration (denoted by * in Table 4) are replicates of previous data sets. Replicates are beneficial for the analysis to understand the natural variability of the process and quantify whether deviations from the model predictions are likely attributed to lack of fit of the model or natural variability. Moreover, the values of some of the variables in the actual test runs differ from the values provided in the test plan due to challenges with precisely setting the prescribed input levels in the plant. For example, issues with controlling the $\mathrm{CO}_{2}$ concentration in the flue gas resulted in considerable deviation between the test plan and data values. There was also some deviation in the lean loading values,
which can be expected since the reboiler steam flowrate was directly manipulated in lieu of lean loading. This deviation is likely attributed to the propagation of discrepancy between the actual process and the process model for the individual unit operations (e.g. absorber, heat exchanger, stripper). Fig. 6 demonstrates that the deterministic model accurately predicts the $\mathrm{CO}_{2}$ capture percentage for the data set, with an average percentage error of $5.2 \pm 4.6 \%$.

Though it was originally planned to implement the Bayesian inference stage of SDoE with a smaller initial data set of 3-5 runs, with more frequent updates of the parameter distributions and the test plan, this was ultimately determined to be impractical due to timing constraints. The SDoE approach is flexible in that it can accommodate different sized stages depending on operational constraints that need to be considered. The change to a larger initial experiment size was driven by logistical issues including, but not limited to, development of an accurate surrogate model, problems with obtaining data for the first few test cases, and the time requirements for plant operators to implement the updated test plan. Therefore, the full set of data shown in Table 4 was incorporated into the Bayesian inference methodology, in which

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3caa1e19-e3fa-41ca-9661-b80d49821925-13.jpg?height=994&width=1401&top_left_y=191&top_left_x=333}
\captionsetup{labelformat=empty}
\caption{Fig. 9. Estimated marginal PDFs for prior and posterior distributions of mass transfer and hydraulics model parameters updated through incorporating $\mathrm{CO}_{2}$ capture percentage data into Bayesian inference for second iteration.}
\end{figure}
the data were used to update the distributions of the hydraulics and mass transfer model parameters while keeping the uncertainty of the thermodynamics model parameters constant. The thermodynamics model parameter values were treated as constant distributions because these distributions were previously quantified rigorously through a Bayesian inference procedure using property data (VLE, heat capacity, heat of absorption) [10]. Since these parameters are theoretically scaleindependent, updating the parameters' distributions through use of plant data would essentially amount to treating them as fitting parameters, which should be avoided. For the remaining parameters, which are process-dependent, the distributions were estimated in previous work [11] although with data collected at a smaller scale. These distributions can be refined through the pilot-scale data collected in this test campaign, and are treated as parameters of variable uncertainty in this work. The updated parameter distributions for the mass transfer and hydraulics models are given in Fig. 7; the names ascribed to the parameters match those defined in Aspen Plus. The parameter ARVAL is the coefficient in the interfacial area model for the absorber column packing, $\mathrm{DFACT} / \mathrm{CO}_{2}$ is the exponential temperature dependence on the diffusivity of $\mathrm{CO}_{2}$ in the aqueous MEA solution, and HURVAL represents liquid holdup model parameters. It is shown that the effect of Bayesian inference on the marginal PDF varies widely by parameter. For example, the parameter ranges (regions in which the probability density is not approximately zero) of DFACT/CO2 ${ }_{2}$ and HURVAL/1 remain about the same although the probability density shifts within the ranges, resulting in updated estimates of the most likely values (modes) of these parameters, which are the values of highest probability density. For HURVAL/2, the parameter range becomes much narrower as a result of the Bayesian inference, indicating that the new data support the conclusion that the uncertainty in the parameter was overestimated in the prior distribution. The most dramatic effect is in the interfacial area parameter (ARVAL/2), for which the region of appreciable probability density essentially shifts to a higher value, suggesting that the new process data have provided evidence that the parameter value was underestimated in the original model.

The effect of the updated parameter distributions on the stochastic model prediction of $\mathrm{CO}_{2}$ capture is shown in Fig. 8 in terms of $95 \%$ confidence intervals calculated for the $\mathrm{CO}_{2}$ capture prediction. For the full input space, the average percentage of reduction in the $95 \%$ confidence intervals for $\mathrm{CO}_{2}$ capture was $32.8 \pm 11.6 \%$, and this value was $37.0 \pm 8.4 \%$ for the set of test conditions for which data were collected. Overall, the average uncertainty of $\mathrm{CO}_{2}$ capture percentage prediction in the candidate set decreases from $5.6 \pm 1.0 \%$ to $3.8 \pm 1.1 \%$ as a result of the first iteration of SDoE.

\subsection*{3.4. Second iteration of sequential design of experiments}

After updating the parameter distributions, the revised stochastic model was used as a basis for an additional iteration of SDoE. A significant time period (approximately 3 h ) was required to develop the new test plan after obtaining the posterior distributions of the parameters. Due to the limited remaining time available in the campaign, only three sets of data were obtained during the second iteration (see Table 5). The criterion used in the second stage of the SDoE was to focus on regions with the largest uncertainty for predicting the $\mathrm{CO}_{2}$ capture percentage (based on the previously discussed concept of G-optimality). A new candidate set was constructed from the largest $20 \%$ of the estimated $95 \%$ confidence intervals. From this candidate set, three runs were selected that provided good augmentation of the original space filling design. More detailed plant data are given in Appendix A.

As in the first iteration of SDoE, these data sets were incorporated into the Bayesian framework and new distributions for the mass transfer and hydraulics model parameters were obtained. The marginal PDFs for each parameter are shown in Figs. 9, and 10 shows the effect of the updated parameter distributions on the calculated confidence intervals of $\mathrm{CO}_{2}$ capture for the candidate points as well as the points at which experimental data were collected.

Note that in Figs. 9 and 10, the 'prior' distribution corresponds exactly to the 'posterior' distribution from the first round of SDoE, due to the sequential nature of the procedure. Over the full input space, the

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3caa1e19-e3fa-41ca-9661-b80d49821925-14.jpg?height=1347&width=1686&top_left_y=180&top_left_x=198}
\captionsetup{labelformat=empty}
\caption{Fig. 10. Effect of second iteration update in stochastic model on model prediction uncertainty ( $95 \%$ confidence interval widths for $\mathrm{CO}_{2}$ capture) for (A) the entire input space and (B) points for which data were collected.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3caa1e19-e3fa-41ca-9661-b80d49821925-14.jpg?height=639&width=1234&top_left_y=1644&top_left_x=409}
\captionsetup{labelformat=empty}
\caption{Fig. 11. Parity plot for $\mathrm{CO}_{2}$ capture percentage in absorber (configured with 3 beds including intercooling) for complete set of NCCC data.}
\end{figure}
average reduction in uncertainty of $\mathrm{CO}_{2}$ capture prediction for the second iteration was $26.7 \pm 24.1 \%$ and the average reduction was $64.9 \pm 7.6 \%$ for the points in which data were collected in the second round of SDoE. For the overall SDoE procedure, the estimated reduction in the uncertainty of predicted $\mathrm{CO}_{2}$ capture percentage, calculated by
comparing the prior uncertainty from the first iteration to the posterior uncertainty for the second iteration, is $52.7 \pm 11.8 \%$ across the full input space. Overall, the average uncertainty of $\mathrm{CO}_{2}$ capture percentage prediction in the candidate set decreases from $3.8 \pm 1.1 \%$ to $2.6 \pm 0.5 \%$ as a result of the second iteration of SDoE.

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 6
Results of one bed absorber test.}
\begin{tabular}{|l|l|l|l|l|l|l|}
\hline \multirow[t]{2}{*}{Data No.} & \multirow[t]{2}{*}{Lean Solvent Flowrate (kg/h)} & \multirow[t]{2}{*}{Flue Gas Flowrate (kg/h)} & \multirow[t]{2}{*}{Lean Solvent Loading (mol CO2/ mol MEA)} & \multirow[t]{2}{*}{Flue Gas $\mathrm{CO}_{2}$ Mole Fraction} & \multicolumn{2}{|l|}{$\mathrm{CO}_{2}$ Capture Percentage} \\
\hline & & & & & Data & Model \\
\hline 1C & 6185 & 1997 & 0.15 & 0.077 & 97.1 & 95.4 \\
\hline 2C & 7765 & 2499 & 0.20 & 0.077 & 92.3 & 87.6 \\
\hline 3C & 7517 & 2013 & 0.25 & 0.093 & 89.5 & 84.0 \\
\hline 4C & 6160 & 1500 & 0.25 & 0.108 & 88.9 & 87.6 \\
\hline 5C & 5237 & 1498 & 0.26 & 0.077 & 86.4 & 87.3 \\
\hline 6C & 7665 & 2700 & 0.31 & 0.077 & 60.2 & 58.8 \\
\hline 7C & 5414 & 1000 & 0.34 & 0.100 & 76.4 & 78.8 \\
\hline
\end{tabular}
\end{table}

A parity plot is given in Fig. 11 for all data, including the previous 2014 campaign and both iterations of this campaign, collected while operating the absorber with three beds of packing and with the intercoolers in use. The fit of the deterministic model is generally good throughout a large range of operating conditions. The average percentage error for the $\mathrm{CO}_{2}$ capture prediction is $4.8 \pm 5.8 \%$ for all data, and the values for the individual 2014 and 2017 campaigns are $4.7 \pm 7.6 \%$ and $4.9 \pm 4.1 \%$, respectively. From this figure, it is apparent that the prevalence of clustering in the $\mathrm{CO}_{2}$ capture percentage data was reduced for the 2017 campaign designed using the SDoE approach in comparison with the 2014 campaign, during which a traditional space-filling approach was used to select the points for data collection.

\subsection*{3.5. Exploration of the design space: Alternative process configurations}

Since the absorber at the NCCC pilot plant contains three beds, with a packing height of 6 m in each bed, separated by intercooling stages and multiple solvent inlets, it can be operated with five different configurations:
- Three beds (18 m total packing height) with intercooling
- Three beds without intercooling
- Two beds (12 m total packing height) with intercooling
- Two beds without intercooling
- One bed (6 m total packing height)

Although the SDoE portion of this work was implemented with all three beds and intercooling, additional data were collected for the other configurations in order to assess the overall validity of the model. During the 2014 MEA campaign, very few runs were completed for absorber configurations other than three beds with intercooling, the default configuration at NCCC. Although the SDoE procedure implemented during the 2017 campaign also used this default absorber configuration, additional data were collected for absorber operation with one bed and two beds without intercooling. These configurations were ultimately chosen due to the prioritization of analyzing the effect of packing height on absorber performance. The remaining configurations, three beds without intercooling and two beds with intercooling,
were not considered in this work due to the limited availability of time. The points for which data were collected were chosen using a spacefilling approach with a minimax criterion for the same ranges for the input variables and constraints on the deterministic model estimate of $\mathrm{CO}_{2}$ capture percentage ( $50-95 \%$ ). The data obtained for one bed and two bed, along with the corresponding model prediction for $\mathrm{CO}_{2}$ capture percentage, are presented in Tables 6 and 7, respectively, along with a parity plot in Fig. 12. The detailed plant data are also given in Appendix A.

For these tests, the model fit the $\mathrm{CO}_{2}$ capture percentage data with average percentage error of $3.1 \pm 2.1 \%$ and $2.7 \pm 2.6 \%$ for operation with one and two beds, respectively. This provides additional support for the overall predictability of this process model at varying scale, especially considering the lack of available data from the 2014 NCCC campaign for operation with one or two beds.

The data collected during the portion of the campaign in which the absorber was operated with one or two beds were not considered when updating the parametric distributions through uncertainty quantification. However, the effect of the Bayesian update of the parameters can also be demonstrated for the absorber model with one or two beds. New deterministic values for the mass transfer and hydraulics parameters are calculated as the mean parameter values from the posterior distribution obtained at the end of the second round of SDoE, which is shown in Fig. 9. The comparison of the original and updated deterministic values of these parameters is shown in Table 8.

The results in Table 8 demonstrate the degree to which the parameter estimates are modified as a result of updating the deterministic values from the original values obtained from maximum likelihood estimation in previous work [11] to the mean values of the final posterior distributions obtained here. The model predictions for these data are re-calculated with the Aspen Plus rate-based absorber model using the updated parameter values, and the parity plot is given in Fig. 13.

With the updated model, the average percentage error for the data not used in the SDoE procedure (one and two bed absorber cases) decreased from $2.9 \pm 2.1 \%$ to $2.7 \pm 1.6 \%$, representing a marginal improvement in the model. This demonstrates that the overall process model predicts well throughout a wide range of absorber packing height, or variation in the number of beds used for gas-liquid contacting, despite the primary focus on the configuration with three beds and intercooling used with the SDoE procedure.

\section*{4. Discussion}

This work represents a first demonstration of executing Bayesian SDoE for a continuously operating pilot plant campaign. Before this methodology can become common practice in pilot testing, foundational capabilities, workflows, and access to fundamental data need to be improved. Quantification of parametric uncertainty, which is often neglected in process systems engineering applications due to its computational expense, is required before executing the SDoE procedure described in this work. This uncertainty quantification requires availability of relevant submodel data, such as VLE data for thermodynamic models and wetted-wall column data for mass transfer models. Without such data, the process for obtaining initial estimates of parameter distributions becomes more subjective and complicated. Currently, the

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 7
Results of two bed absorber test.}
\begin{tabular}{|l|l|l|l|l|l|l|}
\hline \multirow[t]{2}{*}{Data No.} & \multirow[t]{2}{*}{Lean Solvent Flowrate (kg/h)} & \multirow[t]{2}{*}{Flue Gas Flowrate (kg/h)} & \multirow[t]{2}{*}{Lean Solvent Loading (mol CO2/mol MEA)} & \multirow[t]{2}{*}{Flue Gas CO2 Mole Fraction} & \multicolumn{2}{|l|}{$\mathrm{CO}_{2}$ Capture Percentage} \\
\hline & & & & & Data & Model \\
\hline 1D & 4912 & 1500 & 0.30 & 0.100 & 77.8 & 80.1 \\
\hline 2D & 4600 & 2000 & 0.20 & 0.117 & 80.5 & 81.2 \\
\hline 3D & 9534 & 2502 & 0.30 & 0.093 & 87.0 & 81.5 \\
\hline 4D & 4733 & 1966 & 0.20 & 0.079 & 96.4 & 96.9 \\
\hline
\end{tabular}
\end{table}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3caa1e19-e3fa-41ca-9661-b80d49821925-16.jpg?height=751&width=1233&top_left_y=193&top_left_x=417}
\captionsetup{labelformat=empty}
\caption{Fig. 12. Parity plot for model prediction of $\mathrm{CO}_{2}$ capture percentage and experimental data for cases in which absorber is operated with one or two beds.}
\end{figure}

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 8
Comparison of original deterministic parameter values with posterior means obtained from SDoE.}
\begin{tabular}{|l|l|l|l|}
\hline & Original Deterministic Value & Posterior Mean & Percent Change \\
\hline ARVAL/2 & 1.42245 & 1.5375 & 8.1 \\
\hline DFACT/CO2 & $4.56 \mathrm{e}-10$ & $4.8792 \mathrm{e}-10$ & 7.0 \\
\hline HURVAL/1 & 11.45 & 11.335 & -1.0 \\
\hline HURVAL/2 & 0.6471 & 0.5212 & -19.5 \\
\hline
\end{tabular}
\end{table}
hierarchical processes of using Bayesian inference to update the distributions of process-dependent parameters (e.g. mass transfer, hydraulics) while incorporating fixed uncertainty in other parameters (e.g. physical property models) is time consuming, generally requiring up to one day to complete. The algorithm for re-evaluating the model uncertainty prediction throughout the full output space of interest and using the new information in the selection of an updated test plan can take approximately two additional hours to complete. Finally, the development of adequate surrogate models is particularly challenging, particularly when representing an output variable (e.g. $\mathrm{CO}_{2}$ capture
percentage in absorber) as a function of both input variables and model parameters over wide ranges. Although the MARS method was ultimately chosen for this work, future work should evaluate alternate surrogate modeling techniques in order to improve the accuracy of the SDoE procedure.

In this work, the uncertainty in the output variables ( $\mathrm{CO}_{2}$ capture percentage in the example used in this work) is taken into account in the likelihood function used in the Bayesian inference (Eq. (5)) under the assumption that they follow a Gaussian distribution. Appendix A includes information on the estimated uncertainty of the variables in the data collected during the test campaign, and it is suggested that the uncertainty in these variables may be considered negligible in this case. Moreover, another simplification made in this work is that the forms of the submodels considered (thermodynamics, mass transfer, interfacial area, hydraulics) are fixed during the Bayesian inference procedure while only the distributions of the parameters are updated in light of the experimental data. Inclusion of model form uncertainty in the Bayesian inference is another enhancement to the SDoE methodology that will be considered in future work.

Since the SDoE executed in this work was focused on reduction of

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3caa1e19-e3fa-41ca-9661-b80d49821925-16.jpg?height=737&width=1247&top_left_y=1802&top_left_x=409}
\captionsetup{labelformat=empty}
\caption{Fig. 13. Parity plot for one and two-bed absorber $\mathrm{CO}_{2}$ capture percentage with updated deterministic model.}
\end{figure}
uncertainty in the mass transfer and hydraulics submodels, this paper was mostly focused on the absorption section of the MEA process. However, Appendix B has been included for a discussion of the stripper performance and comparison of the process model and experimental data. There are potential operational issues associated with the solvent regeneration section, including heat loss and maldistribution effects in the stripper column, that have not been characterized in the existing process model. For the full data set, the deterministic model generally underpredicts the reboiler duty requirement in the stripper, and more detailed analysis is given in the appendix. Appendix C also shows a comparison of the column temperature profiles predicted by the deterministic model to the experimental data. It is shown that the model matches the absorber temperature profiles with more consistent accuracy compared to the stripper temperature profiles. Therefore, further work on more rigorous uncertainty analysis of performance of the stripper column at NCCC is recommended.

The SDoE methodology is currently being implemented as a module in FOQUS (Framework for Optimization and Quantification of Uncertainty and Surrogates) [66,67], one of the tools available in the open-source CCSI Toolset. The authors are targeting a more streamlined version of the methodology that will enable its widespread implementation in chemical process systems beyond the scope of carbon capture applications.

\section*{5. Conclusions}

A methodology for incorporating Bayesian design of experiments into the execution of a pilot-scale test campaign for solvent-based $\mathrm{CO}_{2}$ capture systems has been developed and demonstrated for an aqueous MEA system at NCCC. Although only two iterations of the SDoE process were performed due to limitations on the amount of time available for the test campaign, the use of SDoE was shown to be effective for reducing the uncertainty in the stochastic model's estimate of $\mathrm{CO}_{2}$ capture percentage in the aqueous MEA process. The initial model predicted $\mathrm{CO}_{2}$ capture percentage with an average uncertainty of $5.6 \pm 1.0 \%$, and this value was reduced to $3.8 \pm 1.1 \%$ and $2.6 \pm 0.5 \%$, respectively, at the end of the first and second rounds of SDoE. This demonstrates the capability of this methodology for reducing the uncertainty in a stochastic model, despite the fact that the original model was quite accurate. It was also determined that the model prediction was improved for alternative process configurations (e.g. absorber operation with one or two beds and no intercooling), despite the fact that no data for these configurations were included in the SDoE. The ability to quantify and reduce uncertainty in process models is essential for reducing technical risk for scale-up of new technologies, and improved models can also facilitate improved techno-economic analyses of new technologies. While it may be questionable whether the SDoE procedure is needed for a system of low uncertainty such as aqueous MEA, especially considering the substantial experimental and computational time requirement for its execution, the main purpose of this work was to demonstrate its ability to reduce parametric uncertainty for a baseline system. The capability of the SDoE procedure to facilitate reduction of uncertainty in a model of a well-studied $\mathrm{CO}_{2}$ capture system such as aqueous MEA demonstrates potential for application of this methodology to accelerating the development and modeling of novel capture
systems.
Future work will focus on further application of the sequential design of experiments methodology, specifically for pilot scale testing of novel $\mathrm{CO}_{2}$ capture technologies. Although the example given in this work was based mostly on model prediction uncertainty for the absorber in an initial effort to benchmark the methodology, the technique can be extended to quantification and reduction of uncertainty in multiple outputs, including energy performance and economic variables. Additional work should also include quantification of the effect of uncertainty on the techno-economic analyses of $\mathrm{CO}_{2}$ capture processes. This would allow for inclusion of risk analysis in the process equipment design and provide some insight into the degree of uncertainty reduction required to reduce equipment sizes, or capital cost, by some amount. Techno-economic analyses with UQ would also be a useful tool for comparing novel $\mathrm{CO}_{2}$ capture processes against a baseline (e.g. aqueous MEA) in consideration of the trade-off between predicted process performance and model precision.

\section*{Author contribution}

All authors contributed equally to this paper.

\section*{Declaration of Competing Interest}

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

\section*{Acknowledgements}

This research was conducted through the Carbon Capture Simulation for Industry Impact (CCSI2), funded through the Carbon Capture Program of U.S. Department of Energy's Office of Fossil Energy. A portion of this work was funded by Lawrence Berkeley Laboratory through contract\# 7210843. This project was supported in part by an appointment to the Science Education Programs at National Energy Technology Laboratory (NETL), administered by ORAU through the U.S. Department of Energy Oak Ridge Institute for Science and Education.

\section*{Disclaimer}

This paper was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor any agency thereof, nor any of their employees, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness of any information, apparatus, product, or process disclosed, or represents that its use would not infringe privately owned rights. Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

\section*{Appendix A. Plant data}

The plant data collected for this work are summarized in Tables $\mathrm{A} 1-\mathrm{A} 3$, respectively for the absorber, lean/rich heat exchanger, and stripper sections, in greater detail than was given in the results section of this work. For each data case, the data are presented in terms of average values of the variables over the specific steady-state period. Since the $\mathrm{CO}_{2}$ capture percentage in the absorber is used as the output variable when using Bayesian inference to update the model parameters, estimates for the variance (or standard deviation) of this variable for each data case are required (see Eq. (5)) and are therefore also included in the table. Unavailable measurements are denoted as 'NA' in Tables A1-A3. For the other variables for which measurements are available in increments of one minute, the average values of the standard deviation calculated from the 29 data points are presented in Table A4. Liquid composition measurements ( $\mathrm{CO}_{2}$ loading and MEA fraction) are not included in this table because they are taken not from continuous measurements but rather from samples taken with a frequency of approximately $1-2 \mathrm{~h}$, and only a single value is available for most

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table A1
Data obtained during 2017 MEA campaign at NCCC (Absorber-Part 1).}
\begin{tabular}{|l|l|l|l|l|l|l|l|}
\hline Case & Lean Solvent Flowrate (kg/h) & Inlet Flue Gas Flowrate (kg/h) & Lean Solvent Temperature ( ${ }^{\circ} \mathrm{C}$ ) & Inlet Flue Gas Temperature ( ${ }^{\circ} \mathrm{C}$ ) & Absorber Top Pressure (kPa) & Inlet Flue Gas Pressure (kPa) & Lean Solvent CO2 Loading ( $\mathrm{mol} \mathrm{CO} 2 / \mathrm{mol}$ MEA) \\
\hline 1A & 8180 & 3000 & 44.1 & 43.4 & 109.4 & 115.2 & 0.24 \\
\hline 2A & 7130 & 2690 & 44.9 & 45.2 & 109.5 & 114.6 & 0.25 \\
\hline 3A & 3354 & 1500 & 45.6 & 43.3 & 107.9 & 110.1 & 0.24 \\
\hline 4A & 3600 & 3000 & 45.6 & 43.4 & 110.2 & 113.6 & 0.19 \\
\hline 5A & 3380 & 2750 & 45.5 & 45.7 & 109.8 & 112.8 & 0.20 \\
\hline 6A & 3130 & 1750 & 45.1 & 43.4 & 108.2 & 109.9 & 0.31 \\
\hline 7A & 4730 & 2255 & 45.1 & 43.5 & 109.2 & 111.8 & 0.23 \\
\hline 8A & 3230 & 2240 & 44.8 & 42.8 & 109.2 & 110.9 & 0.24 \\
\hline 9A & 3224 & 2245 & 45.9 & 43.4 & 108.8 & 111.6 & 0.14 \\
\hline 10A & 7980 & 2492 & 44.8 & 43.1 & 109.0 & 114.5 & 0.32 \\
\hline 11A & 3016 & 2761 & 45.8 & 42.9 & 109.6 & 111.9 & 0.16 \\
\hline 12A & 4170 & 2920 & 46.4 & 45.1 & 109.7 & 114.0 & 0.14 \\
\hline 13A & 6910 & 2680 & 44.9 & 44.6 & 109.3 & 114.9 & 0.26 \\
\hline 14A & 6505 & 2500 & 44.8 & 43.5 & 109.2 & 112.9 & 0.31 \\
\hline 15A & 8000 & 2494 & 44.9 & 42.9 & 109.1 & 114.2 & 0.32 \\
\hline 1B & 7959 & 2497 & 44.4 & 43.7 & 109.7 & 115.2 & 0.30 \\
\hline 2B & 9871 & 2746 & 44.7 & 45.7 & 109.5 & 117.0 & 0.30 \\
\hline 3B & 11,412 & 2748 & 44.4 & 47.3 & 109.2 & 113.0 & 0.30 \\
\hline 1C & 6185 & 1997 & NA & 43.3 & 109.1 & 110.7 & 0.15 \\
\hline 2C & 7765 & 2499 & NA & 43.1 & 109.6 & 112.5 & 0.20 \\
\hline 3C & 7517 & 2013 & NA & 43.6 & 109.5 & 110.9 & 0.25 \\
\hline 4C & 6160 & 1500 & 44.9 & 43.3 & 107.7 & 110.4 & 0.25 \\
\hline 5C & 5237 & 1498 & 45.0 & 43.3 & 108.4 & 108.7 & 0.26 \\
\hline 6C & 7665 & 2700 & 44.8 & 43.4 & 109.7 & 112.4 & 0.31 \\
\hline 7C & 5414 & 1000 & 44.7 & 43.3 & 107.3 & 107.7 & 0.34 \\
\hline 1D & 4912 & 1500 & 44.8 & 43.4 & 107.8 & 109.8 & 0.30 \\
\hline 2D & 4600 & 2000 & 45.4 & 43.9 & 108.7 & 110.7 & 0.20 \\
\hline 3D & 9534 & 2502 & NA & 44.9 & 110.0 & 114.3 & 0.30 \\
\hline 4D & 4733 & 1966 & 46.3 & 43.1 & 108.6 & 110.9 & 0.20 \\
\hline
\end{tabular}
\end{table}

Data obtained during 2017 MEA campaign at NCCC (Absorber-Part 2).

\begin{tabular}{|l|l|l|l|l|l|l|l|}
\hline Case & Lean Solvent MEA Mass Fraction ${ }^{\mathrm{a}}$ & Inlet Flue Gas CO2 Mole Fraction ${ }^{\mathrm{b}}$ & Inlet Flue Gas $\mathrm{O}_{2}$ Mole Fraction & Rich Solvent Flowrate (kg/h) & Rich Solvent Temperature ( ${ }^{\circ} \mathrm{C}$ ) & Outlet Flue Gas Temperature ( ${ }^{\circ} \mathrm{C}$ ) & Rich Solvent Pressure (Absorber Outlet) (kPa) \\
\hline 1A & 0.30 & 0.082 & 0.111 & 8682 & 48.9 & 42.7 & 115.2 \\
\hline 2A & 0.31 & 0.100 & 0.087 & NA & 47.9 & 50.9 & 115.0 \\
\hline 3A & 0.31 & 0.108 & 0.073 & 3641 & 46.3 & 59.4 & 110.8 \\
\hline 4A & 0.31 & 0.076 & 0.111 & 3863 & 46.4 & 56.6 & 113.6 \\
\hline 5A & 0.31 & 0.107 & 0.074 & 3495 & 48.7 & 57.2 & 113.0 \\
\hline 6A & 0.30 & 0.076 & 0.110 & 3366 & 45.2 & 51.8 & 110.3 \\
\hline 7A & 0.30 & 0.109 & 0.071 & 5022 & 46.1 & 57.8 & 112.4 \\
\hline 8A & 0.31 & 0.107 & 0.073 & 3628 & 46.0 & 57.5 & 111.7 \\
\hline 9A & 0.31 & 0.108 & 0.073 & 3526 & 46.1 & 61.5 & 112.1 \\
\hline 10A & 0.30 & 0.109 & 0.072 & 8505 & 46.0 & 51.2 & 114.7 \\
\hline 11A & 0.31 & 0.096 & 0.086 & 3220 & 46.0 & 57.9 & 112.4 \\
\hline 12A & 0.32 & 0.107 & 0.073 & 4407 & 47.8 & 62.5 & 113.7 \\
\hline 13A & 0.30 & 0.108 & 0.073 & 7354 & 46.9 & 55.8 & 115.0 \\
\hline 14A & 0.31 & 0.108 & 0.072 & 6872 & 46.0 & 51.9 & 113.3 \\
\hline 15A & 0.30 & 0.108 & 0.072 & 8436 & 45.8 & 51.1 & 114.6 \\
\hline 1B & 0.30 & 0.077 & 0.110 & 8767 & 48.5 & 44.7 & 116.5 \\
\hline 2B & 0.30 & 0.088 & 0.098 & 10,716 & 50.5 & 43.7 & 117.4 \\
\hline 3B & 0.30 & 0.108 & 0.071 & 12,528 & 50.1 & 43.4 & 113.2 \\
\hline 1C & 0.30 & 0.077 & 0.108 & 6830 & 55.0 & 64.3 & 111.8 \\
\hline 2C & 0.30 & 0.077 & 0.109 & 8366 & 53.9 & 63.8 & 112.9 \\
\hline 3C & 0.30 & 0.093 & 0.090 & 7765 & 56.4 & 65.3 & 112.3 \\
\hline 4C & 0.30 & 0.108 & 0.075 & 6954 & 57.9 & 50.9 & 109.4 \\
\hline 5C & 0.30 & 0.077 & 0.110 & 5976 & 55.9 & 49.4 & 109.8 \\
\hline 6C & 0.31 & 0.077 & 0.109 & 8356 & 50.5 & 50.4 & 112.5 \\
\hline 7C & 0.29 & 0.100 & 0.091 & 6127 & 53.9 & 41.9 & 108.5 \\
\hline 1D & 0.30 & 0.100 & 0.091 & 5600 & 51.6 & 55.5 & 111.0 \\
\hline 2D & 0.30 & 0.117 & 0.071 & 5244 & 48.5 & 61.7 & 111.7 \\
\hline 3D & 0.30 & 0.093 & 0.090 & 10,078 & 54.7 & 65.8 & 115.5 \\
\hline 4D & 0.30 & 0.079 & 0.107 & 5411 & 51.4 & 59.4 & 111.8 \\
\hline
\end{tabular}

Data obtained during 2017 MEA campaign at NCCC (Absorber-Part 3).

\begin{tabular}{|l|l|l|l|l|l|l|l|}
\hline Case & Rich Solvent Pressure (After Pump ${ }^{\mathrm{c}}$ ) (kPa) & Rich Solvent CO2 Loading ( $\mathrm{mol} \mathrm{CO}_{2} / \mathrm{mol}$ MEA) & Rich Solvent MEA Mass Fraction & Outlet Flue Gas $\mathrm{CO}_{2}$ Mole Fraction & Outlet Flue Gas $\mathrm{O}_{2}$ Mole Fraction & Number of Beds (Intercoolers) Used & $\mathrm{CO}_{2}$ Capture Percentage (Average ± Standard Deviation) \\
\hline 1A & 764.8 & 0.46 & 0.32 & 0.001 & 0.121 & 3 (2) & $97.5 \pm 0.3$ \\
\hline 2A & 716.7 & 0.45 & 0.29 & 0.008 & 0.097 & 3 (2) & $93.4 \pm 0.3$ \\
\hline 3A & 598.7 & 0.46 & 0.36 & 0.026 & 0.081 & 3 (2) & $81.1 \pm 0.6$ \\
\hline
\end{tabular}

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table A1 (continued)}
\begin{tabular}{|l|l|l|l|l|l|l|l|}
\hline \multicolumn{8}{|l|}{Data obtained during 2017 MEA campaign at NCCC (Absorber-Part 3).} \\
\hline Case & Rich Solvent Pressure (After Pump ${ }^{\mathrm{c}}$ ) (kPa) & Rich Solvent CO2 Loading (mol CO2/mol MEA) & Rich Solvent MEA Mass Fraction & Outlet Flue Gas $\mathrm{CO}_{2}$ Mole Fraction & Outlet Flue Gas $\mathrm{O}_{2}$ Mole Fraction & Number of Beds (Intercoolers) Used & $\mathrm{CO}_{2}$ Capture Percentage (Average ± Standard Deviation) \\
\hline 4A & 601.8 & 0.46 & 0.32 & 0.027 & 0.118 & 3 (2) & $70.6 \pm 1.1$ \\
\hline 5A & 587.4 & 0.42 & 0.33 & 0.062 & 0.078 & 3 (2) & $53.9 \pm 2.6$ \\
\hline 6A & 591.5 & 0.46 & 0.30 & 0.044 & 0.115 & 3 (2) & $51.7 \pm 0.9$ \\
\hline 7A & 634.8 & 0.46 & 0.32 & 0.037 & 0.079 & 3 (2) & $72.5 \pm 1.0$ \\
\hline 8A & 578.3 & 0.46 & 0.31 & 0.057 & 0.078 & 3 (2) & $56.3 \pm 1.8$ \\
\hline 9A & 578.4 & 0.46 & 0.31 & 0.034 & 0.080 & 3 (2) & $74.2 \pm 0.6$ \\
\hline 10A & 756.2 & 0.47 & 0.30 & 0.026 & 0.080 & 3 (2) & $79.9 \pm 0.6$ \\
\hline 11A & 573.9 & 0.45 & 0.31 & 0.046 & 0.090 & 3 (2) & $60.5 \pm 1.7$ \\
\hline 12A & 603.3 & 0.46 & 0.31 & 0.033 & 0.080 & 3 (2) & $75.4 \pm 0.8$ \\
\hline 13A & 704.6 & 0.47 & 0.32 & 0.026 & 0.080 & 3 (2) & $80.6 \pm 0.6$ \\
\hline 14A & 687.1 & 0.47 & 0.31 & 0.055 & 0.077 & 3 (2) & $57.8 \pm 0.7$ \\
\hline 15A & 753.5 & 0.48 & 0.30 & 0.032 & 0.079 & 3 (2) & $76.8 \pm 1.4$ \\
\hline 1B & 835.2 & 0.48 & 0.29 & 0.003 & 0.120 & 3 (2) & $96.1 \pm 0.1$ \\
\hline 2B & 898.4 & 0.47 & 0.28 & 0.003 & 0.108 & 3 (2) & $97.7 \pm 0.5$ \\
\hline 3B & 961.2 & 0.50 & NA & 0.007 & 0.081 & 3 (2) & $94.8 \pm 0.1$ \\
\hline 1C & 672.2 & 0.36 & 0.32 & 0.003 & 0.118 & 1 (0) & $97.1 \pm 0.7$ \\
\hline 2C & 734.5 & 0.38 & 0.33 & 0.007 & 0.118 & 1 (0) & $92.3 \pm 0.5$ \\
\hline 3C & 723.7 & 0.47 & 0.31 & 0.012 & 0.099 & 1 (0) & $89.5 \pm 1.2$ \\
\hline 4C & 677.1 & 0.42 & 0.31 & 0.014 & 0.084 & 1 (0) & $88.9 \pm 0.8$ \\
\hline 5C & 643.9 & 0.42 & 0.30 & 0.013 & 0.118 & 1 (0) & $86.4 \pm 0.6$ \\
\hline 6C & 734.7 & 0.39 & 0.31 & 0.037 & 0.115 & 1 (0) & $60.2 \pm 0.5$ \\
\hline 7C & 649.6 & 0.43 & 0.30 & 0.030 & 0.098 & 1 (0) & $76.4 \pm 2.2$ \\
\hline 1D & 635.0 & 0.39 & 0.31 & 0.024 & 0.099 & 2 (0) & $78.4 \pm 2.2$ \\
\hline 2D & 628.7 & 0.47 & 0.30 & 0.026 & 0.080 & 2 (0) & $80.5 \pm 0.2$ \\
\hline 3D & 834.5 & 0.42 & 0.29 & 0.015 & 0.099 & 2 (0) & $87.0 \pm 0.9$ \\
\hline 4D & 630.7 & 0.43 & 0.29 & 0.003 & 0.117 & 2 (0) & $96.4 \pm 0.5$ \\
\hline
\end{tabular}
\end{table}
${ }^{\mathrm{a}}$ Solvent MEA fraction is represented on a $\mathrm{CO}_{2}$-free basis.
${ }^{\mathrm{b}}$ Flue gas composition is presented on a $\mathrm{H}_{2} \mathrm{O}$-free (dry) basis throughout this paper. Values for the $\mathrm{CO}_{2}$ and $\mathrm{O}_{2}$ mole fraction are presented, and the balance is $\mathrm{N}_{2}$. The $\mathrm{H}_{2} \mathrm{O}$ content of the inlet flue gas stream may be reasonably estimated by assuming that the flue gas is saturated with $\mathrm{H}_{2} \mathrm{O}$ at the specified values of temeprature and pressure (both given in Part 1 of Table A1).
${ }^{\mathrm{c}}$ Labeled as 'Rich Solvent Pump' in Fig. 2.

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table A2
Data obtained during 2017 MEA campaign at NCCC (Lean/Rich Heat Exchanger).}
\begin{tabular}{|l|l|l|l|l|}
\hline \multirow[t]{3}{*}{Case} & Rich Solvent & Rich Solvent & Lean Solvent & Lean Solvent \\
\hline & Inlet & Outlet & Inlet & Outlet \\
\hline & Temperature $\left({ }^{\circ} \mathrm{C}\right)$ & Temperature $\left({ }^{\circ} \mathrm{C}\right)$ & Temperature $\left({ }^{\circ} \mathrm{C}\right)$ & Temperature $\left({ }^{\circ} \mathrm{C}\right)$ \\
\hline 1A & 49.7 & 108.8 & 115.6 & 52.5 \\
\hline 2A & 48.8 & 109.9 & 116.7 & 51.5 \\
\hline 3A & 47.2 & 108.6 & 116.1 & 49.5 \\
\hline 4A & 47.1 & 109.6 & 117.5 & 49.3 \\
\hline 5A & 49.5 & 109.5 & 117.4 & 51.5 \\
\hline 6A & 45.9 & 104.2 & 110.4 & 48.1 \\
\hline 7A & 46.9 & 108.8 & 115.9 & 49.5 \\
\hline 8A & 46.9 & 109.8 & 118.0 & 49.2 \\
\hline 9A & 47.0 & 109.0 & 118.3 & 48.7 \\
\hline 10A & 46.9 & 108.0 & 114.3 & 50.0 \\
\hline 11A & 46.8 & 109.5 & 118.1 & 48.9 \\
\hline 12A & 48.6 & 110.4 & 119.8 & 50.2 \\
\hline 13A & 47.8 & 109.8 & 116.3 & 50.6 \\
\hline 14A & 46.9 & 106.0 & 112.0 & 49.7 \\
\hline 15A & 46.6 & 107.5 & 114.1 & 49.6 \\
\hline 1B & 49.4 & 108.3 & 114.6 & 52.3 \\
\hline 2B & 51.4 & 109.1 & 115.4 & 54.4 \\
\hline 3B & 50.9 & 108.8 & 115.0 & 54.3 \\
\hline 1C & 55.9 & 112.5 & 119.8 & 57.7 \\
\hline 2C & 54.8 & 112.0 & 118.6 & 57.2 \\
\hline 3C & 57.3 & 111.5 & 117.5 & 59.6 \\
\hline 4C & 59.0 & 110.8 & 116.4 & 60.9 \\
\hline 5C & 56.8 & 109.8 & 115.2 & 58.9 \\
\hline 6C & 51.5 & 107.1 & 112.9 & 54.3 \\
\hline 7C & 54.7 & 105.2 & 110.3 & 56.7 \\
\hline 1D & 52.5 & 107.6 & 113.5 & 54.8 \\
\hline 2D & 49.3 & 109.8 & 117.1 & 51.5 \\
\hline 3D & 55.6 & 110.2 & 116.2 & 58.3 \\
\hline 4D & 52.3 & 110.4 & 117.3 & 54.3 \\
\hline
\end{tabular}
\end{table}

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table A3
Data obtained during 2017 MEA campaign at NCCC (Stripper).}
\begin{tabular}{|l|l|l|l|l|l|l|}
\hline Case & Inlet Rich Solvent Temperature $\left({ }^{\circ} \mathrm{C}\right)$ & Rich Solvent Inlet Pressure (kPa) & Stripper Top Pressure (kPa) & Reboiler Temperature ( ${ }^{\circ} \mathrm{C}$ ) & Lean Solvent Pressure (After Pump ${ }^{\mathrm{a}}$ ) (kPa) & Reboiler Duty (kW) \\
\hline 1A & 102.6 & 446.0 & 180.1 & 119.4 & 280.9 & 420 \\
\hline 2A & 101.4 & 445.8 & 182.6 & 120.5 & 273.6 & 437 \\
\hline 3A & 99.2 & 447.2 & 179.6 & 118.9 & 274.1 & 202 \\
\hline 4A & 100.6 & 445.4 & 179.9 & 119.6 & 273.5 & 253 \\
\hline 5A & 100.2 & 441.8 & 179.7 & 119.4 & 272.6 & 234 \\
\hline 6A & 97.4 & 449.7 & 179.0 & 115.3 & 273.2 & 118 \\
\hline 7A & 99.8 & 448.7 & 180.1 & 119.0 & 273.9 & 335 \\
\hline 8A & 99.9 & 446.7 & 180.1 & 120.2 & 271.8 & 366 \\
\hline 9A & 100.0 & 446.2 & 179.7 & 120.4 & 274.5 & 354 \\
\hline 10A & 99.6 & 447.0 & 179.9 & 120.9 & 276.6 & 412 \\
\hline 11A & 100.1 & 445.3 & 179.8 & 120.2 & 272.8 & 330 \\
\hline 12A & 101.0 & 447.1 & 185.4 & 123.9 & 308.1 & 484 \\
\hline 13A & 100.8 & 445.2 & 184.4 & 122.0 & 274.0 & 423 \\
\hline 14A & 98.5 & 446.6 & 180.5 & 118.8 & 273.8 & 317 \\
\hline 15A & 99.4 & 446.0 & 180.3 & 120.8 & 277.7 & 409 \\
\hline 1B & 101.6 & 446.1 & 180.2 & 118.2 & 273.8 & 313 \\
\hline 2B & 103.0 & 446.7 & 190.9 & 119.2 & 276.4 & 388 \\
\hline 3B & 102.1 & 446.4 & 186.7 & 118.0 & 292.1 & 458 \\
\hline 1C & 111.8 & 446.0 & 186.6 & 121.9 & 276.0 & 431 \\
\hline 2C & 110.2 & 446.3 & 181.4 & 120.6 & 311.1 & 428 \\
\hline 3C & 108.4 & 446.4 & 180.7 & 119.9 & 287.4 & 352 \\
\hline 4C & 106.9 & 446.0 & 181.9 & 119.5 & 273.8 & 299 \\
\hline 5C & 106.6 & 445.8 & 181.7 & 118.3 & 274.0 & 285 \\
\hline 6C & 103.0 & 446.7 & 182.2 & 118.3 & 273.4 & 288 \\
\hline 7C & 101.9 & 446.5 & 181.5 & 115.7 & 273.6 & 198 \\
\hline 1D & 101.7 & 444.7 & 179.5 & 116.6 & 272.8 & 189 \\
\hline 2D & 101.4 & 446.4 & 179.9 & 118.9 & 273.7 & 271 \\
\hline 3D & 105.2 & 446.3 & 181.2 & 119.5 & 273.7 & 370 \\
\hline 4D & 104.9 & 446.1 & 179.9 & 119.0 & 273.8 & 278 \\
\hline
\end{tabular}
\end{table}
${ }^{\mathrm{a}}$ Labeled as 'Lean Solvent Pump' in Fig. 2.

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table A4
Mean and standard deviation values for process variables calculated for full set of 29 data points collected during 2017 MEA test campaign at NCCC.}
\begin{tabular}{|l|l|l|}
\hline Variable & Mean & Standard Deviation \\
\hline \multicolumn{3}{|l|}{Absorber} \\
\hline Lean Solvent Flowrate (kg/h) & $6052 \pm 2288$ & $33 \pm 16$ \\
\hline Inlet Flue Gas Flowrate (kg/h) & $2292 \pm 530$ & $37 \pm 23$ \\
\hline Lean Solvent Temperature ( ${ }^{\circ} \mathrm{C}$ ) & $45.1 \pm 0.6$ & $0.3 \pm 0.5$ \\
\hline Inlet Flue Gas Temperature ( ${ }^{\circ} \mathrm{C}$ ) & $43.8 \pm 1.1$ & $0.2 \pm 0.3$ \\
\hline Absorber Top Pressure (kPa) & $109.1 \pm 0.7$ & $0.2 \pm 0.2$ \\
\hline Inlet Flue Gas Pressure (kPa) & $112.3 \pm 2.2$ & $0.2 \pm 0.1$ \\
\hline Inlet Flue Gas $\mathrm{CO}_{2}$ Mole Fraction & $0.096 \pm 0.014$ & $0.003 \pm 0.002$ \\
\hline Inlet Flue Gas $\mathrm{O}_{2}$ Mole Fraction & $0.088 \pm 0.016$ & $0.003 \pm 0.002$ \\
\hline Rich Solvent Flowrate (kg/h) & $6526 \pm 2484$ & $191 \pm 76$ \\
\hline Rich Solvent Temperature ( ${ }^{\circ} \mathrm{C}$ ) & $49.6 \pm 3.8$ & $0.2 \pm 0.2$ \\
\hline Outlet Flue Gas Temperature ( ${ }^{\circ} \mathrm{C}$ ) & $54.7 \pm 7.1$ & $0.3 \pm 0.3$ \\
\hline Rich Solvent Pressure (Absorber Outlet) (kPa) & $112.8 \pm 2.1$ & $0.2 \pm 0.2$ \\
\hline Rich Solvent Pressure (After Pump) (kPa) & $689.3 \pm 101.0$ & $11.7 \pm 5.7$ \\
\hline Outlet Flue Gas CO2 Mole Fraction & $0.025 \pm 0.017$ & $0.001 \pm 0.001$ \\
\hline Inlet Flue Gas $\mathrm{O}_{2}$ Mole Fraction & $0.096 \pm 0.017$ & $0.003 \pm 0.002$ \\
\hline $\mathrm{CO}_{2}$ Capture Percentage & $79.4 \pm 14.4$ & $0.9 \pm 0.7$ \\
\hline \multicolumn{3}{|l|}{Lean/Rich Heat Exchanger} \\
\hline Rich Solvent Inlet Temperature ( ${ }^{\circ} \mathrm{C}$ ) & $50.5 \pm 3.8$ & $0.2 \pm 0.2$ \\
\hline Rich Solvent Outlet Temperature ( ${ }^{\circ} \mathrm{C}$ ) & $109.1 \pm 1.9$ & $0.3 \pm 0.4$ \\
\hline Lean Solvent Inlet Temperature ( ${ }^{\circ} \mathrm{C}$ ) & $115.9 \pm 2.4$ & $0.1 \pm 0.2$ \\
\hline Lean Solvent Outlet Temperature ( ${ }^{\circ} \mathrm{C}$ ) & $52.9 \pm 3.8$ & $0.3 \pm 0.3$ \\
\hline \multicolumn{3}{|l|}{Stripper} \\
\hline Rich Solvent Inlet Temperature ( ${ }^{\circ} \mathrm{C}$ ) & $102.4 \pm 3.5$ & $0.2 \pm 0.2$ \\
\hline Rich Solvent Inlet Pressure (kPa) & $446.2 \pm 1.3$ & $6.6 \pm 2.7$ \\
\hline Stripper Top Pressure (kPa) & $181.6 \pm 2.7$ & $0.6 \pm 0.7$ \\
\hline Reboiler Temperature ( ${ }^{\circ} \mathrm{C}$ ) & $119.4 \pm 1.8$ & $0.2 \pm 0.2$ \\
\hline Lean Solvent Pressure (After Pump) (kPa) & $227.8 \pm 9.8$ & $3.8 \pm 3.0$ \\
\hline Reboiler Duty (kW) & $333 \pm 91$ & $2.6 \pm 4.2$ \\
\hline
\end{tabular}
\end{table}
data cases. More information on the estimated uncertainty of these measurements is included in our previous work for modeling the MEA process at NCCC [12].

The results presented in Table A4 give some insight into the noise present in the steady-state data for many of the key process variables. The results in the 'Mean' column of Table A4 demonstrate that there is considerable variation only in the variables that were intended to be manipulated in the test campaign. Other variables, including the column pressures and temperatures of the solvent and flue gas streams into the absorber, were essentially held constant throughout the test campaign. Comparing the 'Mean' and 'Standard Deviation' columns, it is apparent that the variation among the process variables within the steady-state test runs is relatively small. The variables with the largest ratio of standard deviation to mean, or highest level of noise, are those measurements for the dry flue gas composition (inlet and outlet $\mathrm{CO}_{2}$ and $\mathrm{O}_{2}$ mole fractions), for which the ratio is within the range $0.03-0.04$. The ratio of standard deviation to mean for the rich solvent flowrate is also approximately 0.03 ; this ratio is less than 0.02 for all other variables included here. Therefore, it is shown that it should be reasonable to neglect the input variable uncertainty in the Bayesian inference algorithm that is used to update the parametric uncertainty, as suggested earlier in this paper.

\section*{Appendix B. Stripper performance}

Although the main scope of the design of experiments executed in this work was to reduce the uncertainty in parameters related to mass transfer and hydraulics, which are primarily relevant for the absorption section of the model, data were also collected for the solvent regeneration process and are compared to the model here. The metric used to compare the model predictions with the data is the specific reboiler duty (SRD), or the ratio of the reboiler duty required as input to the stripper to the amount of $\mathrm{CO}_{2}$ capture. For a given test case, the required reboiler duty is calculated by matching the $\mathrm{CO}_{2}$ loading in the lean solvent stream exiting the bottom of the stripper with the value specified in the lean solvent inlet to the absorber.

The resulting values of SRD are shown in Table B1 and compared to the experimental values. For each data set, the average and standard deviation values are reported for the SRD in order to provide some insight into the amount of noise in the data. This information is also reported in a parity plot in Fig. B1.

As shown in Fig. B1, the process model underpredicts the SRD for most of the test cases, and the average percentage error in the SRD for the full campaign is $11.29 \pm 10.97 \%$. For data sets in which the SRD is below $4 \mathrm{MJ} / \mathrm{kg} \mathrm{CO}_{2}$, the model prediction matches the data more accurately, with an average percentage error of $4.2 \pm 3.9 \%$. This suggests that the stripper model is relatively accurate in the region of desirable operability (low SRD), but there are still some operational issues throughout the full operating space that are not being captured in the model, potentially related to heat loss and maldistribution effects in the stripper. As suggested in the discussion section of this paper, there is potential for future work related to the reconciliation of the discrepancy in the stripper model prediction.

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table B1
Comparison of data with model predictions of specific reboiler duty (SRD).}
\begin{tabular}{|l|l|l|l|}
\hline \multirow[t]{2}{*}{Case} & \multicolumn{2}{|l|}{Specific Reboiler Duty (Data) [MJ/kg CO2]} & \multirow[t]{2}{*}{Specific Reboiler Duty (Model) [MJ/kg CO2 ]} \\
\hline & Average & Standard Deviation & \\
\hline 1A & 4.451 & 0.078 & 4.379 \\
\hline 2A & 3.900 & 0.229 & 4.293 \\
\hline 3A & 3.708 & 0.107 & 3.718 \\
\hline 4A & 3.681 & 0.106 & 3.672 \\
\hline 5A & 3.556 & 0.293 & 3.585 \\
\hline 6A & 4.052 & 0.127 & 3.991 \\
\hline 7A & 4.297 & 0.182 & 3.718 \\
\hline 8A & 6.470 & 0.281 & 3.647 \\
\hline 9A & 4.735 & 0.127 & 3.673 \\
\hline 10A & 4.583 & 0.071 & 3.961 \\
\hline 11A & 4.441 & 0.497 & 3.575 \\
\hline 12A & 4.789 & 0.096 & 3.575 \\
\hline 13A & 4.352 & 0.096 & 3.817 \\
\hline 14A & 4.889 & 0.120 & 3.936 \\
\hline 15A & 4.737 & 0.103 & 3.977 \\
\hline 1B & 3.976 & 0.087 & 4.234 \\
\hline 2B & 3.927 & 0.105 & 4.237 \\
\hline 3B & 3.903 & 0.055 & 4.146 \\
\hline 1C & 6.787 & 0.195 & 6.752 \\
\hline 2C & 5.656 & 0.138 & 5.567 \\
\hline 3C & 5.082 & 0.284 & 4.820 \\
\hline 4C & 4.993 & 0.151 & 4.594 \\
\hline 5C & 6.728 & 0.137 & 4.934 \\
\hline 6C & 5.409 & 0.107 & 4.789 \\
\hline 7C & 7.125 & 0.355 & 4.686 \\
\hline 1D & 4.226 & 0.375 & 3.840 \\
\hline 2D & 3.770 & 0.052 & 3.706 \\
\hline 3D & 4.406 & 0.229 & 4.326 \\
\hline 4D & 4.478 & 0.131 & 4.188 \\
\hline
\end{tabular}
\end{table}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3caa1e19-e3fa-41ca-9661-b80d49821925-22.jpg?height=670&width=819&top_left_y=198&top_left_x=624}
\captionsetup{labelformat=empty}
\caption{Fig. B1. Parity plot for specific reboiler duty required for stripper for all data sets collected at NCCC in Summer 2017 campaign.}
\end{figure}

\section*{Appendix C. Temperature profiles}

For further model validation, experimental and model-generated temperature profiles for all data sets generated in this work are presented here. Absorber temperature profiles are given in Figs. C1-C5 and the stripper temperature profiles in Figs. C6-C10. In all figures, the abscissa (relative column position) represents the position in the packing from top to bottom, so that values of 0 and 1 indicate the top and bottom, respectively, of the column packing. For the absorber column, the value of 0 represents the top of the first section of packing that is used (e.g. the top of the highest section of packing for the cases with three beds in use and the top of the lowest section of packing for the cases with only one bed in use). The temperature profile data are also tabulated in Tables C1-C3 for the absorber column and Table C. 4 for the stripper column.

As shown in Figs. C1-C5, the process model generally captures the shape of the absorber temperature profiles accurately with some exception (e.g. Case 7C). The shape of the absorber temperature profile is heavily dependent on several process variables, which have been varied widely in this test campaign, including the L/G ratio, lean $\mathrm{CO}_{2}$ loading, number of beds in use, and inclusion of solvent intercooling. As shown in Figs. C6-C10, the model does not accurately predict the stripper temperature profile with the same consistency as demonstrated for the absorber. This is consistent with the findings in Appendix B, and suggests potential for future work in improving the modeling of the stripper at NCCC.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3caa1e19-e3fa-41ca-9661-b80d49821925-22.jpg?height=1036&width=1726&top_left_y=1494&top_left_x=165}
\captionsetup{labelformat=empty}
\caption{Fig. C1. Comparison of model and data temperature profiles for absorber column (Cases 1-6A).}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3caa1e19-e3fa-41ca-9661-b80d49821925-23.jpg?height=1022&width=1696&top_left_y=189&top_left_x=180}
\captionsetup{labelformat=empty}
\caption{Fig. C.2. Comparison of model and data temperature profiles for absorber column (Cases 7-12A).}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3caa1e19-e3fa-41ca-9661-b80d49821925-23.jpg?height=1028&width=1743&top_left_y=1455&top_left_x=161}
\captionsetup{labelformat=empty}
\caption{Fig. C3. Comparison of model and data temperature profiles for absorber column (Cases 13-15A and 1-3B).}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3caa1e19-e3fa-41ca-9661-b80d49821925-24.jpg?height=1026&width=1730&top_left_y=191&top_left_x=172}
\captionsetup{labelformat=empty}
\caption{Fig. C4. Comparison of model and data temperature profiles for absorber column (Cases 1-6C).}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3caa1e19-e3fa-41ca-9661-b80d49821925-24.jpg?height=944&width=1741&top_left_y=1388&top_left_x=163}
\captionsetup{labelformat=empty}
\caption{Fig. C5. Comparison of model and data temperature profiles for absorber column (Cases 7C and 1-4D).}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3caa1e19-e3fa-41ca-9661-b80d49821925-25.jpg?height=1022&width=1730&top_left_y=189&top_left_x=172}
\captionsetup{labelformat=empty}
\caption{Fig. C6. Comparison of model and data temperature profiles for stripper column (Cases 1-6A).}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3caa1e19-e3fa-41ca-9661-b80d49821925-25.jpg?height=1015&width=1739&top_left_y=1386&top_left_x=163}
\captionsetup{labelformat=empty}
\caption{Fig. C7. Comparison of model and data temperature profiles for stripper column (Cases 7-12A).}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3caa1e19-e3fa-41ca-9661-b80d49821925-26.jpg?height=1026&width=1743&top_left_y=193&top_left_x=163}
\captionsetup{labelformat=empty}
\caption{Fig. C8. Comparison of model and data temperature profiles for stripper column (Cases 13-15A and 1-3B).}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3caa1e19-e3fa-41ca-9661-b80d49821925-26.jpg?height=1009&width=1733&top_left_y=1433&top_left_x=167}
\captionsetup{labelformat=empty}
\caption{Fig. C9. Comparison of model and data temperature profiles for stripper column (Cases 1-6C).}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3caa1e19-e3fa-41ca-9661-b80d49821925-27.jpg?height=949&width=1743&top_left_y=191&top_left_x=163}
\captionsetup{labelformat=empty}
\caption{Fig. C10. Comparison of model and data temperature profiles for stripper column (Cases 7C and 1-4D).}
\end{figure}

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table C1
Tabulated profiles of absorber temperature ( ${ }^{\circ} \mathrm{C}$ ) [Three Bed Cases].}
\begin{tabular}{|l|l|l|l|l|l|l|l|l|l|l|l|l|}
\hline \multirow[t]{2}{*}{Case No.} & \multicolumn{12}{|l|}{Relative Column Position} \\
\hline & 0.07 & 0.13 & 0.20 & 0.27 & 0.47 & 0.53 & 0.60 & 0.73 & 0.80 & 0.87 & 0.93 & 1.00 \\
\hline 1A & 43.9 & 46.8 & 49.1 & 49.2 & 54.4 & 60.0 & 54.5 & 58.2 & 58.2 & 54.6 & 51.5 & 51.6 \\
\hline 2A & 60.9 & 60.5 & 54.7 & 63.3 & 48.8 & 55.3 & 48.4 & 45.2 & 48.0 & 47.9 & 48.3 & 50.5 \\
\hline 3A & 68.8 & 63.1 & 51.8 & 53.4 & 46.9 & 46.5 & 46.3 & 45.2 & 45.4 & 45.1 & 45.5 & 46.5 \\
\hline 4A & 61.3 & 56.2 & 44.8 & 49.9 & 44.1 & 43.0 & 42.5 & 45.0 & 45.3 & 45.2 & 45.6 & 46.5 \\
\hline 5A & 59.9 & 53.0 & 43.6 & 47.0 & 45.5 & 44.9 & 44.6 & 47.2 & 47.5 & 47.4 & 47.9 & 48.9 \\
\hline 6A & 57.4 & 53.2 & 39.4 & 47.8 & 36.2 & 39.0 & 38.9 & 40.0 & 42.6 & 42.6 & 43.5 & 47.4 \\
\hline 7A & 66.2 & 60.9 & 49.4 & 53.7 & 48.5 & 46.1 & 45.9 & 44.4 & 44.9 & 45.2 & 45.5 & 46.4 \\
\hline 8A & 62.0 & 54.5 & 44.5 & 48.4 & 44.5 & 43.4 & 42.6 & 44.4 & 44.8 & 44.5 & 45.0 & 45.9 \\
\hline 9A & 66.1 & 58.2 & 43.8 & 49.4 & 42.7 & 42.4 & 42.2 & 44.3 & 44.9 & 44.6 & 45.1 & 46.3 \\
\hline 10A & 63.5 & 60.4 & 56.6 & 62.9 & 53.0 & 51.9 & 49.8 & 45.1 & 45.8 & 45.9 & 46.0 & 46.4 \\
\hline 11A & 60.7 & 53.2 & 42.2 & 46.2 & 43.4 & 42.7 & 42.4 & 44.7 & 45.0 & 44.9 & 45.2 & 46.0 \\
\hline 12A & 67.1 & 60.5 & 47.1 & 53.1 & 43.0 & 45.0 & 43.7 & 45.1 & 46.3 & 46.8 & 47.2 & 49.9 \\
\hline 13A & 66.8 & 65.2 & 57.2 & 63.3 & 51.5 & 51.5 & 48.2 & 44.2 & 45.8 & 46.3 & 46.5 & 48.5 \\
\hline 14A & 61.4 & 59.5 & 51.1 & 58.7 & 48.7 & 47.3 & 46.2 & 44.2 & 45.0 & 45.0 & 45.2 & 46.0 \\
\hline 15A & 63.2 & 60.3 & 55.9 & 62.5 & 51.9 & 50.9 & 49.1 & 45.1 & 45.7 & 45.7 & 45.7 & 46.2 \\
\hline 1B & 49.0 & 50.9 & 53.2 & 53.2 & 61.4 & 59.4 & 53.6 & 49.7 & 52.2 & 51.7 & 49.7 & 48.9 \\
\hline 2B & 46.3 & 48.0 & 52.4 & 50.9 & 58.2 & 61.4 & 58.3 & 51.5 & 53.7 & 53.5 & 52.5 & 53.1 \\
\hline 3B & 46.8 & 54.7 & 57.5 & 60.4 & 60.5 & 60.2 & 56.3 & 49.7 & 50.5 & 50.7 & 50.4 & 50.4 \\
\hline
\end{tabular}
\end{table}

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table C2
Tabulated profiles of absorber temperature ( ${ }^{\circ} \mathrm{C}$ ) [One Bed Cases].}
\begin{tabular}{|l|l|l|l|l|l|}
\hline \multirow[t]{2}{*}{Case No.} & \multicolumn{5}{|l|}{Relative Column Position} \\
\hline & 0.20 & 0.40 & 0.60 & 0.80 & 1.00 \\
\hline 1C & 73.5 & 73.6 & 69.4 & 60.3 & 55.2 \\
\hline 2C & 73.9 & 72.7 & 67.7 & 58.6 & 54.8 \\
\hline 3C & 74.9 & 73.7 & 68.7 & 60.8 & 57.0 \\
\hline 4C & 70.5 & 71.0 & 68.7 & 58.7 & 59.1 \\
\hline 5C & 64.9 & 66.4 & 64.3 & 55.5 & 57.6 \\
\hline 6C & 62.6 & 63.4 & 58.0 & 52.8 & 52.4 \\
\hline 7C & 51.2 & 58.2 & 54.7 & 51.4 & 54.9 \\
\hline
\end{tabular}
\end{table}

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table C3
Tabulated profiles of absorber temperature ( ${ }^{\circ} \mathrm{C}$ ) [Two Bed Cases].}
\begin{tabular}{|l|l|l|l|l|l|l|l|l|}
\hline \multirow[t]{2}{*}{Case No.} & \multicolumn{8}{|l|}{Relative Column Position} \\
\hline & 0.20 & 0.30 & 0.40 & 0.60 & 0.70 & 0.80 & 0.90 & 1.00 \\
\hline 1D & 70.3 & 69.0 & 68.8 & 63.5 & 58.3 & 57.6 & 55.8 & 51.8 \\
\hline 2D & 70.5 & 65.1 & 61.3 & 57.2 & 52.9 & 50.5 & 49.8 & 48.4 \\
\hline 3D & 76.5 & 75.9 & 74.6 & 73.0 & 69.8 & 64.2 & 59.8 & 55.6 \\
\hline 4D & 72.6 & 72.4 & 71.8 & 70.2 & 65.3 & 59.0 & 57.0 & 51.2 \\
\hline
\end{tabular}
\end{table}

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table C4
Tabulated profiles of stripper temperature ( ${ }^{\circ} \mathrm{C}$ ).}
\begin{tabular}{|l|l|l|l|l|l|l|l|l|l|l|}
\hline \multirow[t]{2}{*}{Case No.} & \multicolumn{10}{|l|}{Relative Column Position} \\
\hline & 0.10 & 0.20 & 0.30 & 0.40 & 0.50 & 0.60 & 0.70 & 0.80 & 0.90 & 1.00 (Reboiler) \\
\hline 1A & 99.0 & 100.7 & 98.6 & 97.4 & 106.6 & 109.2 & 111.9 & 111.6 & 116.8 & 119.4 \\
\hline 2A & 95.6 & 99.4 & 98.5 & 97.9 & 109.5 & 107.2 & 108.1 & 111.1 & 118.2 & 120.5 \\
\hline 3A & 98.0 & 97.9 & 98.3 & 98.1 & 97.9 & 97.7 & 101.0 & 114.1 & 116.9 & 118.9 \\
\hline 4A & 99.4 & 99.1 & 99.6 & 99.4 & 100.8 & 106.5 & 111.3 & 116.4 & 118.2 & 119.6 \\
\hline 5A & 99.0 & 98.6 & 99.1 & 98.8 & 99.2 & 104.2 & 110.0 & 116.1 & 118.1 & 119.4 \\
\hline 6A & 93.6 & 95.5 & 94.7 & 93.0 & 95.6 & 95.2 & 95.1 & 104.2 & 111.4 & 115.3 \\
\hline 7A & 98.2 & 98.1 & 98.5 & 96.1 & 98.4 & 98.2 & 101.8 & 114.1 & 117.1 & 119.0 \\
\hline 8A & 106.9 & 110.4 & 112.9 & 113.8 & 114.8 & 115.4 & 116.3 & 117.6 & 119.0 & 120.2 \\
\hline 9A & 110.2 & 112.6 & 114.3 & 114.9 & 115.4 & 115.7 & 116.4 & 117.7 & 119.1 & 120.4 \\
\hline 10A & 95.9 & 98.3 & 98.5 & 98.5 & 98.7 & 99.2 & 110.5 & 112.0 & 114.7 & 120.9 \\
\hline 11A & 103.2 & 107.6 & 111.1 & 112.7 & 113.9 & 114.7 & 115.8 & 117.5 & 119.1 & 120.2 \\
\hline 12A & 111.0 & 111.6 & 112.9 & 112.2 & 116.1 & 116.4 & 116.8 & 117.4 & 121.1 & 123.9 \\
\hline 13A & 99.3 & 99.0 & 99.2 & 99.5 & 104.7 & 108.5 & 107.3 & 115.6 & 117.2 & 122.0 \\
\hline 14A & 97.6 & 97.3 & 97.7 & 97.6 & 97.4 & 98.7 & 102.4 & 111.6 & 112.5 & 118.8 \\
\hline 15A & 95.3 & 97.9 & 98.0 & 98.1 & 98.3 & 99.4 & 110.1 & 111.8 & 114.7 & 120.8 \\
\hline 1B & 100.7 & 100.5 & 100.8 & 100.8 & 101.1 & 101.9 & 107.0 & 113.8 & 115.1 & 118.2 \\
\hline 2B & 100.2 & 101.0 & 101.5 & 99.4 & 101.8 & 105.2 & 109.9 & 114.3 & 115.8 & 119.2 \\
\hline 3B & 101.0 & 100.7 & 101.0 & 101.0 & 101.2 & 103.0 & 104.4 & 114.8 & 115.3 & 118.0 \\
\hline 1C & 114.3 & 114.6 & 114.9 & 115.6 & 117.0 & 117.2 & 117.4 & 118.8 & 120.4 & 121.9 \\
\hline 2C & 110.1 & 109.9 & 111.1 & 112.4 & 114.2 & 114.2 & 114.7 & 117.3 & 119.0 & 120.6 \\
\hline 3C & 107.4 & 107.7 & 108.8 & 109.3 & 111.4 & 112.6 & 113.4 & 116.6 & 118.4 & 120.1 \\
\hline 4C & 106.0 & 107.0 & 108.4 & 109.4 & 113.1 & 112.6 & 112.6 & 113.3 & 117.4 & 119.5 \\
\hline 5C & 102.1 & 104.3 & 105.2 & 104.3 & 105.4 & 105.6 & 108.6 & 109.8 & 115.9 & 118.3 \\
\hline 6C & 101.4 & 101.3 & 102.2 & 100.8 & 101.6 & 101.0 & 100.4 & 112.2 & 113.8 & 118.4 \\
\hline 7C & 98.7 & 99.7 & 98.8 & 96.4 & 99.2 & 99.2 & 101.6 & 104.6 & 112.2 & 115.7 \\
\hline 1D & 100.7 & 100.4 & 100.7 & 100.6 & 100.8 & 108.6 & 111.1 & 112.8 & 114.0 & 116.6 \\
\hline 2D & 100.6 & 101.8 & 107.3 & 109.0 & 112.8 & 114.4 & 114.7 & 116.4 & 117.6 & 118.9 \\
\hline 3D & 104.1 & 104.0 & 104.4 & 104.5 & 105.5 & 110.1 & 111.3 & 115.6 & 116.6 & 119.5 \\
\hline 4D & 104.5 & 106.3 & 109.7 & 110.9 & 113.1 & 114.5 & 114.8 & 116.5 & 117.8 & 119.0 \\
\hline
\end{tabular}
\end{table}

\section*{References}
[1] Metz B, Davidson O, de Coninck H, Loos M, Meyer L, editors. IPCC Special Report on Carbon Dioxide Capture and Storage. Cambridge: Cambridge University Press; 2005.
[2] Figueroa JD, Fout T, Plasynski S, McIlvried H, Srivastava RD. Advances in CO2 capture technology - the U.S. Department of Energy's Carbon Sequestration Program. Int J Greenh Gas Con 2008;2:9-20.
[3] Miller DC, Syamlal M, Mebane DS, Storlie C, Bhattacharyya D, Sahinidis NV, et al. Carbon capture simulation initiative: a case study in multiscale modeling and new challenges. Annu Rev Chem Biomol Eng 2014;5:301-23.
[4] Plevin RJ, O'Hare M, Jones AD, Torn MS, Gibbs HG. Greenhouse gas emissions from biofuels' indirect land use change are uncertain but may be much greater than previously estimated. Environ Sci Technol 2010;44:8015-21.
[5] Kimaev G, Ricardez-Sandoval LA. Multilevel Monte Carlo applied to chemical engineering systems subject to uncertainty. AIChE J 2018;64(5):1651-61.
[6] Whiting W. Effects of uncertainties in thermodynamic data and models on process calculations. J Chem Eng Data 1996;41:935-41.
[7] Mathias PM. Sensitivity of process design to phase equilibrium - a new perturbation method based on the Margules equation. J Chem Eng Data 2014;59(4):1006-15.
[8] Myers RH. Response surface methodology - current status and future directions. J Qual Technol 1999;31(1):30-44.
[9] Morgan JC, Bhattacharyya D, Tong C, Miller DC. Uncertainty quantification of property models: methodology and its application to $\mathrm{CO}_{2}$-loaded aqueous MEA solutions. AIChE J 2015;61(6):1822-39.
[10] Morgan JC, Soares Chinen A, Omell B, Bhattacharyya D, Tong C, Miller DC. Thermodynamic modeling and uncertainty quantification of $\mathrm{CO}_{2}$-loaded aqueous MEA solutions. Chem Eng Sci 2017;168:309-24.
[11] Chinen AS, Morgan JC, Omell B, Bhattacharyya D, Tong C, Miller DC. Development of a rigorous modeling framework for solvent-based $\mathrm{CO}_{2}$ capture. Part 1: Hydraulic and mass transfer models and their uncertainty quantification. Ind Eng Chem Res 2018;57:10448-63.
[12] Morgan JC, Soares Chinen A, Omell B, Bhattacharyya D, Tong C, Miller DC. Development of a rigorous modeling framework for solvent-based $\mathrm{CO}_{2}$ capture. Part 2: Steady-state validation and uncertainty quantification with pilot plant data. Ind Eng Chem Res 2018;57:10464-81.
[13] Cerrillo-Briones IM, Ricardez-Sandoval LA. Robust optimization of a post-combustion $\mathrm{CO}_{2}$ capture absorber column under process uncertainty. Chem Eng Res Des 2019;144:386-96.
[14] Chinen AS, Morgan JC, Omell BP, Bhattacharyya D, Miller DC. Dynamic data reconciliation and model validation of a MEA-based CO2 capture system using pilot plant data. IFAC-PapersOnLine 2016;49(7):639-44.
[15] Chinen AS, Morgan JC, Omell B, Bhattacharyya D, Miller DC. Dynamic data reconciliation and validation of a dynamic model for solvent-based $\mathrm{CO}_{2}$ capture using pilot-plant data. Ind Eng Chem Res 2019;58:1978-93.
[16] Chaloner K, Verdinelli I. Bayesian experimental design: a review. Stat Sci 1995;10(3):237-304.
[17] Scott AJ, Nabifar A, Madhuranthakam CMR, Penlidis A. Bayesian design of experiments applied to a complex polymerization system: nitrile butadiene rubber production in a train of CSTRs. Macromol Theor Simul 2015;24(1):13-27.
[18] Bisetti F, Kim D, Knio O, Long Q, Tempone R. Optimal Bayesian experimental design for priors of compact support with application to shock-tube experiments for combustion kinetics. Int J Num Meth Eng 2016;108:136-55.
[19] Ryan EG, Drovandi CC, Pettitt AN. Fully Bayesian experimental design for pharmacokinetic studies. Entropy 2015;17:1063-89.
[20] Atkinson AC, Bogacka B. Compound D- and $\mathrm{D}_{\mathrm{s}}$-optimum designs for determining the order of a chemical reaction. Technometrics 1997;39(4):347-56.
[21] Chen Q, Paulavičius, Adjiman CS. An optimization framework to combine operable space maximization with design of experiments. AIChE J 2018;64(11):3944-67.
[22] Kreutz C, Timmer J. Systems biology: experimental design. FEBS J 2009;276:923-42.
[23] Solonen A, Haario H, Laine M. Simulation-based optimal design using a response variance criterion. J Comput Graph Stat 2012;21(1):234-52.
[24] Kalyanaraman J, Fan Y, Labreche Y, Lively RP, Kawajiri Y, Realff MJ. Bayesian estimation of parametric uncertainties, quantification and reduction using optimal design of experiments for $\mathrm{CO}_{2}$ adsorption on amine sorbents. Comput Chem Eng 2015;18:376-88.
[25] Konomi B, Karagiannis G, Sarkar A, Sun X, Lin G. Bayesian tree multivariate Gaussian process with adaptive design: application to a carbon capture unit. Technometrics 2014;56(2):145-58.
[26] Soepyan FB, Anderson-Cook CM, Morgan JC, Tong CH, Bhattacharyya D, Omell BP, et al. Sequential design of experiments to maximize learning from carbon capture pilot plant testing. Comput Aided Chem Eng 2018;44:283-8.
[27] Li F, Zhang J, Oko E, Wang M. Modeling of a post-combustion $\mathrm{CO}_{2}$ capture process using neural networks. Fuel 2015;151:156-63.
[28] Sipöcz N, Tobiesen FA, Assadi M. The use of artificial neural network models for $\mathrm{CO}_{2}$ capture plants. Appl Energy 2011;88:2368-76.
[29] Zhou Q, Wu Y, Chan CW, Tontiwachwuthikul P. From neural network to neurofuzzy modeling: applications to the carbon dioxide capture process. Energy Proc 2011;4:2066-73.
[30] Hemmati A, Rashidi H, Hemmati A, Kazemi A. Using rate based simulation, sensitivity analysis and response surface methodology for optimization of an industrial $\mathrm{CO}_{2}$ capture plant. J Nat Gas Sci Eng 2019;62:101-12.
[31] Liu H, Chan C, Tontiwachwuthikul P, Idem R. Analysis of CO2 equilibrium solubility of seven tertiary amine solvents using thermodynamic and ANN models. Fuel 2019;249:61-72.
[32] Mesbah M, Shahsavari S, Soroush E, Rahaei N, Rezakazemi M. Accurate prediction of miscibility of $\mathrm{CO}_{2}$ and supercritical $\mathrm{CO}_{2}$ in ionic liquids using machine learning. J $\mathrm{CO}_{2}$ Util 2018;25:99-107.
[33] Venkatraman V, Alsberg BK. Predicting $\mathrm{CO}_{2}$ capture of ionic liquids using machine learning. J CO2 Util 2017;21:162-8.
[34] Yarveicy H, Ghiasi MM, Mohammadi AH. Performance evaluation of the machine learning approaches in $\mathrm{CO}_{2}$ equilibrium absorption in piperazine aqueous solution. J Mol Liq 2018;255:375-83.
[35] Kim Y, Jang H, Kim J, Lee J. Prediction of storage efficiency on CO2 sequestration in deep saline aquifers using artificial neural network. Appl Energy 2017;185:916-28.
[36] Amundsen TG, $\varnothing$ LE, Eimer DA. Density and viscosity of monoethanolamine + water + carbon dioxide from ( 25 to 80$)^{\circ} \mathrm{C} . \mathrm{J}$ Chem Eng Data 2009;54:3096-100.
[37] Cousins A, Wardhaugh L, Cottrell A. Pilot plant operation for liquid absorptionbased post-combustion $\mathrm{CO}_{2}$ capture. In: Feron PMH, editor. Absorption-based postcombustion capture of carbon dioxide. Woodhead Publishing; 2016. p. 649-84.
[38] Gelowitz D, Supap T, Abdulaziz N, Sema T, Idem R, Tontiwachwuthikul P. Part 8: Post-combustion $\mathrm{CO}_{2}$ capture: pilot plant operation issues. Carbon Manag 2013;4(2):215-31.
[39] Bui M, Tait P, Lucquiaud M, Mac Dowell N. Dynamic operation and modelling of amine-based CO2 capture at pilot scale. Int J Greenh Gas Con 2018;79:134-53.
[40] Bui M, Gunawan I, Verheyen V, Feron P. Dynamic modelling and optimization of flexible operation in post-combustion $\mathrm{CO}_{2}$ capture plants - a review. Comput Chem Eng 2014;61:245-65.
[41] Brigman N, Shah MI, Falk-Pedersen O, Cents T, Smith V, de Cazenove T, et al. Results of amine plant operations from $30 \mathrm{wt} \%$ and $40 \mathrm{wt} \%$ aqueous MEA testing at the $\mathrm{CO}_{2}$ Technology Centre Mongstad. Energy Proc 2014;63:6012-22.
[42] Gjernes E, Pedersen S, Cents T, Watson G, Fostås BF, Shah MI, et al. Results from 30 $\mathrm{wt} \%$ MEA performance testing at the $\mathrm{CO}_{2}$ Technology Centre Mongstad. Energy Proc 2017;114:1146-57.
[43] Faramarzi L, Thimsen D, Hume S, Maxon A, Watson G, Pedersen S, et al. Results from MEA testing at the $\mathrm{CO}_{2}$ Technology Centre Mongstad: verification of baseline results in 2015. Energy Proc 2017;114:1128-45.
[44] Montañés RM, Flø NE, Nord LO. Dynamic process model validation and control of
the amine plant at $\mathrm{CO}_{2}$ Technology Centre Mongstad. Energies 2017;10(10):1527.
[45] Bui M, Flø NE, de Cazenove T, Mac Dowell N. Demonstrating flexible operation of the Technology Centre Mongstad (TCM) CO2 capture plant. Int J Greenh Gas Con 2020;93:102879.
[46] Mangalapally HP, Hasse H. Pilot plant study of post-combustion carbon dioxide capture by reactive absorption: methodology, comparison of different structured packings, and comprehensive results for monoethanolamine. Chem Eng Res Des 2011;89:1216-28.
[47] Notz R, Manalapally HP, Hasse H. Post combustion CO2 capture by reactive absorption: pilot plant description and results of systematic studies with MEA. Int J Greenh Gas Con 2012;6:84-112.
[48] Sønderby TL, Carlsen KB, Fosbøl PL, Kiørboe LG, von Solms N. A new pilot absorber for $\mathrm{CO}_{2}$ capture from flue gases: measuring and modeling capture with MEA solution. Int J Greenh Gas Con 2013;12:181-92.
[49] Dugas R, Alix P, Lemaire E, Broutin P, Rochelle G. Absorber model for $\mathrm{CO}_{2}$ capture by monoethanolamine - application to CASTOR pilot results. Energy Proc 2009;1:103-7.
[50] Moser P, Schmidt S, Sieder G, Garcia H, Stoffregen T. Performance of MEA in a longterm test at the post-combustion capture pilot plant in Niederaussem. Int J Greenh Gas Con 2011;5:620-7.
[51] Zhang Y, Chen H, Chen C-C, Plaza JM, Dugas R, Rochelle GT. Rate-based process modeling study of $\mathrm{CO}_{2}$ capture with aqueous monoethanolamine solution. Ind Eng Chem Res 2009;48:9233-46.
[52] Artanto Y, Jansen J, Pearson P, Do T, Cottrell A, Meuleman E, et al. Performance of MEA and amine-blends in the CSIRO PCC pilot plant at Loy Yang Power in Australia. Fuel 2012;101:264-75.
[53] Tobiesen FA, Svendsen HF, Juliussen O. Experimental validation of a rigorous absorber model for $\mathrm{CO}_{2}$ postcombustion capture. AIChE J 2007;53(4):846-65.
[54] Tobiesen FA, Juliussen O, Svendsen HF. Experimental validation of a rigorous desorber model for $\mathrm{CO}_{2}$ post-combustion capture. Chem Eng Sci 2008;63:2641-56.
[55] Mejdell T, Vassbotn T, Juliussen O, Tobiesen A, Einbu A, Knuutila H, et al. Energy Proc 2011;4:1753-60.
[56] Koller M, Wappel D, Trofaier N, Gronald G. Test results of CO2 spray scrubbing with monoethanolamine. Energy Proc 2011;4:1777-82.
[57] Li K, Cousins A, Yu H, Feron P, Tade M, Luo W, et al. Systematic study of aqueous monoethanolamine-based $\mathrm{CO}_{2}$ capture processes: model development and process improvement. Energy Sci Eng 2016;4(1):23-39.
[58] Tong C. PSUADE Reference Manual (Version 1.7). Livermore CA: Lawrence Livermore National Laboratory; 2015https://github.com/LLNL/psuade/blob/ master/Doc/Manual/PsuadeRefManual.pdf.
[59] Robert C, Casella G. A short history of Markov Chain Monte Carlo: subjective recollections from incomplete data. Stat Sci 2011;26(1):102-15.
[60] Press WH, Teukolsky SA, Vetterling WT, Flannery BP. Modeling of data. Numerical recipes in C: the art of scientific computing. 2nd ed.Cambridge: Cambridge University Press; 1992. p. 656-706.
[61] Joseph VR. Space-filling designs for computer experiments: a review. Qual Eng 2016;28(1):28-35.
[62] Johnson ME, Moore LM, Ylvisaker D. Minimax and maximin distance designs. J Stat Plan Infer 1990;26:131-48.
[63] Myers RH, Montgomery DC, Anderson-Cook CM. Practical design optimality. Response surface methodology: process and product optimization using designed experiments. 4th ed.New York: Wiley; 2016. p. 467-74.
[64] Morton F, Laird R, Northington J. The national carbon capture center: cost-effective test bed for carbon capture R\&D. Energy Proc 2013;37:525-39.
[65] Friedman JH. Multivariate adaptive regression splines. Ann Stat 1991;19:1-141.
[66] Eslick J, Ng B, Gao Q, Tong CH, Sahinidis NV, Miller DC. A framework for optimization and quantification of uncertainty and sensitivity for developing carbon capture systems. Energy Proc 2014;63:1055-63.
[67] Miller DC, Agarwal D, Bhattacharyya D, Boverhof J, Cheah Y-W, Chen Y, et al. Innovative computational tools and models for the design, optimization, and control of carbon capture processes. Comp Aid Ch 2016;38:2391-6.