\title{
Advanced absorber heat integration via heat exchange packings
}

\author{
Thomas Moore ( ) Du Nguyen | Jaisree Iyer | Pratanu Roy | Joshuah K. Stolaroff
}

Lawrence Livermore National Laboratory, Livermore, California

\section*{Correspondence}

Joshuah K. Stolaroff, Lawrence Livermore National Laboratory, 7000 East Ave, Livermore, CA 94551.
Email: stolaroff1@llnl.gov

\section*{Funding information}
U.S. Department of Energy, Grant/Award Numbers: FWP-FEW0225, DE-AC52-07NA27344

\begin{abstract}
A rate-based model of an absorption column was developed and used to analyze several intercooling strategies utilizing "heat exchange packings." These packings are capable of removing heat from the column and transferring it to a cooling fluid within the packing. For absorption of $\mathrm{CO}_{2}$ into aqueous monoethanolamine under industrial conditions, intercooling via heat exchange packings placed along 10-20\% of the column could reduce the column height by $\sim 15 \%$. The height of these columns was close to the minimum theoretical value, calculated by numerically optimizing the temperature profile. Effective intercooling could also be achieved by using the cool, rich solvent as the cooling fluid. This reduces the cooling load and facilitates recovery of waste heat. Heat exchange packings could also be used to redistribute heat within the column, reducing the column height with no net cooling load. However, this approach requires larger heat transfer coefficients than have been experimentally observed.
\end{abstract}

\section*{KEYWORDS}
absorption, environmental engineering, heat transfer, separation techniques

\section*{1 | INTRODUCTION}

If dangerous climate change is to be avoided, anthropogenic emissions of greenhouse gases must be drastically reduced over the next few decades. ${ }^{1}$ A promising means of cutting $\mathrm{CO}_{2}$ emissions from fossil power plants and industrial processes is carbon capture and storage $(\mathrm{CCS}) .^{2,3,4} \mathrm{~A}$ wide range of technologies are available for CCS, utilizing solvents, ${ }^{5}$ solid sorbents, ${ }^{6}$ membranes, ${ }^{7}$ and hybrid materials. ${ }^{8,9,10}$ Solvent-based CCS technologies are the most mature, ${ }^{11}$ with a number of industrial-scale plants in operation globally. ${ }^{12}$ Over the last decade, sophisticated heat integration and unit operation design have decreased the energy requirements for postcombustion capture from coal power plants from around $3.6 \mathrm{GJ} / \mathrm{tCO}_{2}$ in $2010^{13}$ to values approaching $2 \mathrm{GJ} / \mathrm{tCO}_{2},^{14,15}$ resulting in significant reductions in process cost. ${ }^{16}$ It is hoped that the development of advanced, nonaqueous solvents may reduce energy requirements further ${ }^{17,18}$ though this has been disputed in the literature. ${ }^{19}$

Absorption columns are the largest units in a gas sweetening process, and considerable research and development has been directed
toward enhancing their performance via improved packing and process design. ${ }^{20}$ One of the most widely investigated process modifications is absorber intercooling, in which the solvent is cooled as it flows down the column (the most common arrangement, in-and-out cooling, is shown in Figure 1.) Absorber intercooling can improve column performance by increasing the thermodynamic driving force for mass transfer and reducing the magnitude of the temperature bulge. ${ }^{21}$ Intercooling has been investigated for postcombustion capture from coal ${ }^{22}$ and natural gas power plants, ${ }^{23}$ and for a range of solvents including MEA, ${ }^{24}$ pipperazine, ${ }^{25}$ and potassium carbonate. ${ }^{26}$ Oko et al. ${ }^{27}$ recently demonstrated that intercooling is essential for rotating bed absorbers utilizing concentrated ( $70 \mathrm{wt} \%$ ) MEA solutions, and discussed a range of process configurations. For traditional postcombustion carbon capture processes, most studies predict that intercooling is able to reduce the absorption column height by about $10 \%$. While this is not an insignificant improvement, it is possible that a more sophisticated heat integration strategy could result in significantly shorter columns. Furthermore, while traditional intercooling processes are able to remove the large quantities of heat created by
the exothermic reactions occurring in the absorber, they do not typically recover this low grade heat, but instead transfer it to a cooling water stream, increasing the process cooling load.

A promising approach to intensifying heat integration within an absorption column is the use of 3D-printed packings capable of removing heat from the column and transfering it directly into a cooling fluid flowing within the packing itself (see Figure 2). Such packings were recently manufactured by Miramontes et al., ${ }^{28,29}$ who tested their performance in a bench-scale column. Their design was a modified version of the Mellapak 250Y structured packing which contained channels in the packing wall through which a cooling fluid could flow. Heat exchange packings could also be manufactured using

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/7251841f-ade3-439f-839f-850b9c70e3cb-02.jpg?height=918&width=587&top_left_y=804&top_left_x=290}
\captionsetup{labelformat=empty}
\caption{FIGURE 1 Traditional in-and-out intercooling. Part of the solvent stream is removed, cooled, and reinjected back into the column}
\end{figure}
triply periodic minimal surfaces (TPMS), which have shown promising performance in heat exchangers ${ }^{30}$ and membrane reactor systems ${ }^{31}$ (Figure 2(B)). A wide range of TPMS structures may be mathematically defined and created via additive manufacturing techniques, and structures may be created in which two spatially disconnected domains are intertwined together in intimate contact. If such structures are used as packings within an absorption column, one domain could contain the counter-current gas and solvent streams, while a second, smaller domain could contain a heat exchange fluid (Figure 2(C)).

Heat exchange packings have the potential to provide intensified, in situ heat exchange within the column itself, obviating the need for external cooling units, and the very large surface area available for heat transfer (on the order $200 \mathrm{~m}^{2} / \mathrm{m}^{3}$ column) may allow significant heat transfer between streams of similar temperature. This in turn opens the door to a range of novel process configurations, which may incorporate sophisticated heat redistribution within the column, as well as the useful recovery of low grade heat, which would otherwise go to waste.

In this article, a number of process designs which utilize heat exchange packings within an absorber for CCS are modeled, in order to evaluate the improvements in process performance which such packings could provide. A future article will describe the manufacture and experimental characterization of TPMS-based heat exchange packings, but the present work is independent of the heat exchange packing topology. To begin, the development and validation of a ratebased model for a CCS absorber utilizing monoethanolamine (MEA) is described. The model is comparable to that developed by Saimpert et al. ${ }^{32}$ though the model developed in this work is capable of explicitly modeling the flow of cooling fluid within the packing itself. The model is then validated against pilot column data, and the qualitative behavior of the temperature bulge is analyzed. The model is then used to calculate optimal temperature profiles along the length of the column , which minimize the column height. The optimal temperature at each point along the column represents a trade-off between increasing the temperature to improve the reaction kinetics, and decreasing the temperature to improve the absorption driving force. These competing effects result in nontrivial optimal profiles, which have not

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/7251841f-ade3-439f-839f-850b9c70e3cb-02.jpg?height=642&width=1295&top_left_y=1918&top_left_x=157}
\captionsetup{labelformat=empty}
\caption{FIGURE 2 Multiscale schematic of heat exchange packing paradigm. (A) Absorption column containing section of heat exchange, or "active," packing. (B) Gyroid TPMS structure with two spatially disjoint domains. (C) Heat transfer between the absorption domain (containing solvent and gas phases) and the heat exchange domain [Color figure can be viewed at wileyonlinelibrary.com]}
\end{figure}
previously been calculated in the literature. These profiles place an upper limit on the reduction in column height that intercooling can provide. Finally, a range of heat integration strategies which utilize heat exchange packings are modeled and evaluated. Apart from the straightforward idea of flowing cooling water through the heat exchange packing, we also consider the possibility of using the cool, rich solvent as the cooling fluid (eliminating the cooling duty, and also recovering waste heat from the absorber) and the possibility of heat redistribution from hotter regions of the column to cooler regions, which could eliminate the cooling duty present in a traditional process while still reducing the column height.

\section*{2 | MODEL DEVELOPMENT}

A rate based model of an absorber was developed which was capable of simulating capture of $\mathrm{CO}_{2}$ using aqueous MEA. The model was similar to the model presented by Saimpert et al. ${ }^{32}$, which in turn was based on work by Pandya et al., ${ }^{33}$ Tontiwachwuthikul et al., ${ }^{34}$ and Simon et al. ${ }^{35}$ The model was one dimensional, and tracked the temperature and molar flow rates of the gas and solvent phases flowing counter currently along the length of the column. Material and energy balances and empirical rate equations were used to derive a set of ordinary differential equations (ODEs), which, once a sufficient number of boundary conditions had been specified at the top and bottom of the column, were numerically solved using a boundary value problem (BVP) solver.

\section*{2.1 | Mass and energy balances}

Shell balances along the length of absorption column were used to derive ODEs for the flows along the column length. Mass and energy flows included within the model are shown in Figure 3, while relevant physical parameters and correlations are listed in Table 1. It was assumed that
- Only $\mathrm{CO}_{2}$ and $\mathrm{H}_{2} \mathrm{O}$ could transfer between phases.
- Axial and radial dispersion were neglected. This assumption has been made by multiple authors who developed rate-based absorber models for carbon capture. $21,22,32,36,37,38,39,40$ Axial dispersion may be ignored provided the dispersive Bodenstein number is large. Macias-Salinas and Fair ${ }^{41}$ reported Bodenstein numbers on the order of $10-100$ at gas and liquid flowrates relevant for $\mathrm{CO}_{2}$ capture. Under these conditions, dispersive fluxes are much smaller than convective fluxes, and may safely be ignored.
- The total concentration of $\mathrm{CO}_{2}$ in the liquid was tracked, including $\mathrm{CO}_{2}$ bound in chemical form.
- The total equivalent concentration of MEA in the liquid was tracked, including unreacted and reacted MEA in its various ionic forms.
- As the thermal conductivity of the liquid is much greater than that of the gas, heat released or consumed by vaporization or

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/7251841f-ade3-439f-839f-850b9c70e3cb-03.jpg?height=829&width=557&top_left_y=169&top_left_x=1207}
\captionsetup{labelformat=empty}
\caption{FIGURE 3 Local mass and heat transfer within absorber model}
\end{figure}
condensation and liquid-phase chemical reactions were included as source/sink terms for the liquid phase only. ${ }^{21}$
- The enthalpies of vaporization and condensation, the enthalpies of reaction, and the specific heat capacity of the liquid and gas were all treated as constant (see Table 1.) Similar assumptions have been made by several other authors ${ }^{21,39}$. Weiland et al. ${ }^{42}$ and Chiu et al. ${ }^{43}$ showed that the heat capacity of aqueous MEA solutions is only weakly dependent on loading and temperature. Similarly, calorimetric ${ }^{44}$ and vapor-liquid equilibria data ${ }^{45}$ both suggest $\Delta H_{\mathrm{CO}_{2}}^{\text {abs }}$ is relatively constant over the temperature and $\mathrm{CO}_{2}$ loading range present in the absorber.
- The absorber was adiabatic.
- The area for interfacial heat and mass transfer were both equal to the wetted area of the packing.
- The gas phase was an ideal gas.
- The column was isobaric. In an actual column, the pressure drop is typically on the order of $10 \mathrm{kPa},{ }^{46,47}$ and the errors in the predicted mass transfer rates resulting from the isobaric assumption were deemed reasonable for the purposes of this study.

In this model, the state of the absorber was described via the molar flow rates of all gas and liquid species, $G_{i}(z)$ and $L_{i}(z)$ (these are flow rates per unit cross-sectional area, with units $\mathrm{mol} / \mathrm{m}^{2} \mathrm{~s}$ ), as well as the temperature profiles in the gas and liquid phases, $T_{G}(z)$ and $T_{L}(z)$. Under the assumptions described above, the following mass balances may be derived for each of the gas and liquid flow rates:

$$
\begin{equation*}
\frac{\mathrm{d} G_{\mathrm{CO}_{2}}}{\mathrm{dz}}=-a_{\mathrm{w}} N_{\mathrm{CO}_{2}} \tag{1}
\end{equation*}
$$


$$
\begin{equation*}
\frac{d G_{\mathrm{H}_{2} \mathrm{O}}}{d z}=-a_{\mathrm{w}} N_{\mathrm{H}_{2} \mathrm{O}} \tag{2}
\end{equation*}
$$


\begin{table}
\captionsetup{labelformat=empty}
\caption{TABLE 1 Physical properties}
\begin{tabular}{|l|l|l|}
\hline Property & Source & Value/Correlation/Notes \\
\hline Density of MEA & [73] & $1050 \mathrm{~kg} / \mathrm{m}^{3}$ \\
\hline Heat of absorption of $\mathrm{CO}_{2}$ in MEA, $\Delta H_{\mathrm{CO}_{2}}^{\mathrm{abs}}$ & [21] & $-82 \mathrm{~kJ} / \mathrm{mol}$ \\
\hline Heat of condensation of $\mathrm{H}_{2} \mathrm{O}$ in MEA, $\Delta H_{\mathrm{H}_{2} \mathrm{O}}^{\text {cond }}$ & [21] & $-48 \mathrm{~kJ} / \mathrm{mol}$ \\
\hline Diffusivity of $\mathrm{CO}_{2}$ in MEA, $\mathcal{D}_{\mathrm{CO}_{2}}$ & [74] & Equations (2) and (16) of [74] \\
\hline Henry's constant of $\mathrm{CO}_{2}$ in MEA, $\mathscr{H}$ & [75] & Equations (2), (3) of [75] \\
\hline Equilibrium $\mathrm{CO}_{2}$ pressure in MEA, $p_{\mathrm{CO}_{2}}^{\text {eq }}$ & [50] & Equation (11) of [50] \\
\hline Reaction rate constant, $k_{2}$ & [76] & $4.4 \times 10^{8} \exp \left(-\frac{5400}{T(K)}\right) \frac{\mathrm{m}^{3}}{\mathrm{~mol}}$ \\
\hline Heat capacity of gaseous $\mathrm{CO}_{2}$ & [77] & $37 \mathrm{~kJ} / \mathrm{mol} \mathrm{K}$ (at $25^{\circ} \mathrm{C}$ ) \\
\hline Heat capacity of gaseous $\mathrm{N}_{2}$ & [77] & $29 \mathrm{~J} / \mathrm{mol} \mathrm{K}$ (at $25^{\circ} \mathrm{C}$ ) \\
\hline Heat capacity of gaseous $\mathrm{H}_{2} \mathrm{O}$ & [77] & $33.5 \mathrm{~J} / \mathrm{mol} \mathrm{K}$ (at $25^{\circ} \mathrm{C}$ ) \\
\hline Heat capacity of gas mixture, $c_{p}^{G}$ & - & Mole-weighted average of pure gas values at $25^{\circ} \mathrm{C}$ \\
\hline Heat capacity of 30 wt \% MEA at 25\% CO2 loading, $50^{\circ} \mathrm{C}, c_{p}^{L}$ & [42;43] & $83.9 \mathrm{~J} / \mathrm{mol} \mathrm{K}$ \\
\hline Diffusivity of $\mathrm{H}_{2} \mathrm{O}$ in $\mathrm{N}_{2}, D_{\mathrm{G}, \mathrm{H}_{2} \mathrm{O}}$ & Fitted to data from [78] & $22.5 \times 10^{-6}\left(\frac{T_{G}}{273.15 K}\right)^{1.8} \mathrm{m}^{2} \mathrm{~s}^{-1}$ \\
\hline Gas density, $\rho_{\mathrm{G}}$ & - & Ideal gas law \\
\hline Pure gas viscosities & [79] & Sutherland formulas using parameters of [79] \\
\hline Mixed gas viscosity & [80]. cf., [55]. & Equations (13), (14) of [80] \\
\hline Viscosity of pure water & [81] & Equation (4) of [81] \\
\hline Viscosity of MEA & [73] & Equations (2)-(4) of [73] \\
\hline Pure gas thermal conductivities & [77] & \\
\hline Mixed gas thermal conductivity, $\lambda$ & - & Mole-weighted average of pure gas values at $25^{\circ} \mathrm{C}$ \\
\hline Vapor pressure of water, $p_{\mathrm{H}_{2} \mathrm{O}}^{\text {vap }}$ & [77] & Antoine equation based on data of [82] \\
\hline
\end{tabular}
\end{table}

$$
\begin{gather*}
\frac{d G_{\mathrm{N}_{2}}}{\mathrm{dz}}=0  \tag{3}\\
\frac{d L_{\mathrm{CO}_{2}}}{\mathrm{dz}}=-a_{\mathrm{w}} N_{\mathrm{CO}_{2}}  \tag{4}\\
\frac{d L_{\mathrm{H}_{2} \mathrm{O}}}{\mathrm{dz}}=-a_{\mathrm{w}} N_{\mathrm{H}_{2} \mathrm{O}}  \tag{5}\\
\frac{\mathrm{~d} L_{\mathrm{am}}}{\mathrm{dz}}=0 \tag{6}
\end{gather*}
$$


Here $a_{\mathrm{w}}$ is the wetted area of the packing per unit volume of absorber, and $N_{i}$ is the flux of species $i$ from the gas to the liquid. $L_{\mathrm{CO}_{2}}$ and $L_{\mathrm{am}}$ refer to the total equivalent flow of $\mathrm{CO}_{2}$ and MEA, respectively, irrespective of the chemical state these may take within the fluid. The fraction of wetted area was calculated using the correlation of Onda et al. ${ }^{48}$ Methods for calculating $N_{\mathrm{CO}_{2}}$ and $N_{\mathrm{H}_{2} \mathrm{O}}$ are discussed below.

The following energy balances may also be derived from shell balances over the gas and liquid phases:

$$
\begin{gather*}
\frac{\mathrm{d} T_{G}}{\mathrm{~d} z}=-\frac{1}{c_{p}^{G} G}\left(a_{\mathrm{w}} q\right)  \tag{7}\\
\frac{\mathrm{d} T_{L}}{\mathrm{~d} z}=-\frac{1}{c_{p}^{L} L} a_{\mathrm{w}}\left(q-N_{\mathrm{H}_{2} \mathrm{O}} \Delta H_{\mathrm{H}_{2} \mathrm{O}}^{\mathrm{cond}}-N_{\mathrm{CO}_{2}} \Delta H_{\mathrm{CO}_{2}}^{\mathrm{abs}}\right) \tag{8}
\end{gather*}
$$

where $q$ is the flux of heat from the gas to the liquid phase, $c_{P}^{G}$ and $c_{P}^{L}$ are the heat capacities of the gas and liquid, respectively, $\Delta H_{\mathrm{CO}_{2}}^{\mathrm{abs}}$ is the enthalpy of absorption of $\mathrm{CO}_{2}$ in MEA, and $\Delta \mathrm{H}_{\mathrm{H}_{2} \mathrm{O}}^{\text {cond }}$ is the enthalpy of condensation of $\mathrm{H}_{2} \mathrm{O}$ in MEA.

\section*{2.2 | Rate of interphase mass and heat transfer}

\subsection*{2.2.1 | $\mathrm{CO}_{2}$ transfer}

The absorption of $\mathrm{CO}_{2}$ into an aqueous MEA solution was assumed to be liquid-phase controlled and governed by diffusion with a pseudofirst order reaction with MEA. ${ }^{49,24}$ Under these conditions, the $\mathrm{CO}_{2}$ flux is given by:

$$
\begin{equation*}
N_{\mathrm{CO}_{2}}=\sqrt{k_{2} \mathrm{C}_{\mathrm{MEA}} \mathcal{D}_{\mathrm{CO}_{2}}} \mathcal{H}\left(p y_{\mathrm{CO}_{2}}-p_{\mathrm{CO}_{2}}^{\mathrm{eq}}\right) \tag{9}
\end{equation*}
$$

where $k_{2}$ is a second order rate constant, $c_{\text {MEA }}$ is the concentration of unreacted MEA in the solution, $\mathcal{D}_{\mathrm{CO}_{2}}$ is the diffusivity of $\mathrm{CO}_{2}$ in the solvent, $\mathcal{H}$ is the Henry's constant for $\mathrm{CO}_{2}$ in the solvent, $p_{\mathrm{CO}_{2}}^{\mathrm{eq}}$ is the equilibrium partial pressure of $\mathrm{CO}_{2}$ in the solvent, and $p$ is the column pressure (see Table 1 for relevant values and correlations). c ${ }_{\text {MEA }}$ was calculated by assuming that $\mathrm{CO}_{2}$ reacts with MEA according to the following overall reaction:

$$
\begin{equation*}
\mathrm{CO}_{2}+2 \mathrm{MEA} \rightarrow \mathrm{MEACOO}^{-}+\mathrm{MEAH}^{+} \tag{10}
\end{equation*}
$$


The concentration of unreacted MEA was calculated via the following expression,

$$
c_{\text {MEA }}=c_{a m}(1-2 \theta)
$$

where $c_{\mathrm{am}}$ is the total concentration of amines in the solution, and $\theta$ is the fractional loading of $\mathrm{CO}_{2}$ in the solution. ${ }^{50}$ These are in turn calculated from the molar flows within the liquid as follows:

$$
\begin{equation*}
c_{\mathrm{am}}=\frac{L_{\mathrm{am}}}{L / \rho} ; \quad \theta=\frac{L_{\mathrm{CO}_{2}}}{L_{\mathrm{am}}} \tag{12}
\end{equation*}
$$

where $\rho$ is the molar density of the MEA solution.
In recent years a number of authors have developed more sophisticated models, which account for the reaction of $\mathrm{CO}_{2}$ with water and hydroxide ions, and the reversion of the carbamate into a carbonate and protonated amine. ${ }^{32,24}$ However, the simpler approach outlined above has been used by a number of authors, ${ }^{51,49,52}$ who have argued that the reaction of $\mathrm{CO}_{2}$ with $\mathrm{OH}^{-}$may be ignored at low pH , and that the carbamate is sufficiently stable so that, to first approximation, its breakdown may be ignored. These "classical" approximations were deemed sufficiently accurate for the purposes of this work. The principle benefit of this explicit approach is that it allowed for the development of an efficient and stable model which could be used for iterative optimization calculations.

\subsection*{2.2.2 | $\mathrm{H}_{2} \mathrm{O}$ transfer}

Following Saimpert et al., ${ }^{32}$ the evaporation and condensation of water was assumed to be gas phase controlled,

$$
\begin{equation*}
N_{\mathrm{H}_{2} \mathrm{O}}=k_{\mathrm{G}}\left(y_{\mathrm{H}_{2} \mathrm{O}} p-p_{\mathrm{H}_{2} \mathrm{O}}^{\mathrm{eq}}\right) . \tag{13}
\end{equation*}
$$


The gas phase mass transfer coefficient, $k_{G}$, was estimated using the correlation of Billet and Schultes, ${ }^{53}$ using physical property data and correlations provided in Table 1. The partial pressure of water was calculated via Raoults law. ${ }^{54}$

\subsection*{2.2.3 | Heat transfer}

The flux of heat between the two phases was calculated by the following expression,

$$
\begin{equation*}
q=h\left(T_{G}-T_{L}\right) \tag{14}
\end{equation*}
$$

where $h$ is the heat transfer coefficient. It was assumed that heat transfer was gas phase controlled, and the gas phase heat transfer coefficient was calculated via the Chilton-Colburn analogy, ${ }^{55,32}$

$$
\begin{equation*}
h=k_{G}\left(\frac{\rho_{G} \lambda^{2}}{D_{G, \mathrm{H}_{2} \mathrm{O}}^{2}} \frac{c_{p, G}}{M_{W, G}}\right)^{1 / 3} \tag{15}
\end{equation*}
$$

where $\lambda$ is the gas thermal conductivity and $M_{\text {W, } G}$ is the molecular weight of the gas.

\section*{2.3 | Boundary conditions}

In order to fully specify the system, boundary conditions must be supplied at the top and bottom of the column. The most natural approach
is to specify the liquid species flow rates and the liquid temperature at the top of the column:

$$
\begin{equation*}
\left.L_{i}\right|_{z=\mathscr{L}}=L_{i}^{\text {top }} ;\left.\quad T_{L}\right|_{z=\mathscr{L}}=T_{L}^{\text {top }} \tag{16}
\end{equation*}
$$

and the gas species flow rates and the gas temperature at the bottom:

$$
\begin{equation*}
\left.G_{i}\right|_{z=0}=G_{i}^{\text {bot }} ;\left.\quad T_{G}\right|_{z=0}=T_{G}^{\text {bot }} \tag{17}
\end{equation*}
$$

where $\mathscr{L}$ is the length of the column. With these boundary conditions specified, the number of boundary conditions matches the number of ODEs (Equations (1-8)), and the ODEBVP system is fully specified. However, in this work it was necessary to find the column length required for a given degree of capture (e.g., $90 \%$ capture of $\mathrm{CO}_{2}$ ). Hence, in addition to Equations (16) and (17), one more boundary condition was specified:

$$
\begin{equation*}
\left.G_{\mathrm{CO}_{2}}\right|_{z=\mathscr{L}}=G_{\mathrm{CO}_{2}}^{\mathrm{top}}=(1-\alpha) G_{\mathrm{CO}_{2}}^{\mathrm{bot}} \tag{18}
\end{equation*}
$$

where $\alpha$ is the fraction of $\mathrm{CO}_{2}$ removed along the column length (e.g., for $90 \%$ capture, $\alpha=0.9$ ). In order to avoid over specifying the system, the column length, $\mathscr{L}$, was then left unspecified, and was solved for along with the remaining unknown variables within the ODEBVP.

\section*{2.4 | Numerical methods}

The ODEBVP system defined by Equations (1-8) and (16-18), and the various constitutive equations described above, was numerically solved using a multiple-point shooting method ${ }^{56}$ implemented in Julia. ${ }^{57}$ Shooting methods were found to be more numerically efficient than collocation methods (cf., Reference 32) but when a single-point shooting method was used the numerical instability introduced upon converting the ODEBVP to an initial value problem (IVP) ${ }^{56}$ led to poor model stability, particularly when mass transfer pinches became more pronounced near the minimum liquid flow rate. To address this issue, both 10-point and 50-point multiple shooting methods were implemented. The 10-point method was found to be sufficiently stable for the vast majority of applications, and is used in most calculations below. The 50-point method was slower to run, but was capable of simulating even severelypinched systems, and this code was used for the accurate calculation of the minimum liquid flow rate.

The IVP solver used within the shooting method was the Tsit5 algorithm available within the DifferentialEquations.jl package, while the resulting set of nonlinear equations was solved using the trust region algorithm available within the NLSolve.jl package. ${ }^{58}$

\section*{2.5 | Explicit simulation of heat exchange packings}

The adiabatic model described above was modified to incorporate a cooling fluid flowing up the column through a second domain within
the packing. The modified model used an energy balance to predict the temperature of the cooling fluid and the influence of the cooling fluid on the temperatures and flow rates within the gas and solvent phases. In particular, Equation (8) in the adiabatic model was replaced by:

$$
\begin{equation*}
\frac{\mathrm{d} T_{L}}{\mathrm{dz}}=-\frac{1}{c_{p}^{L} L} a_{w}\left(q+q_{\mathrm{wall}}-N_{\mathrm{H}_{2} \mathrm{O}} \Delta H_{\mathrm{H}_{2} \mathrm{O}}^{\text {cond }}-N_{\mathrm{CO}_{2}} \Delta H_{\mathrm{CO}_{2}}^{\text {abs }}\right) \tag{19}
\end{equation*}
$$

where $q_{\text {wall }}$ is the flux of heat from the heat exchange packing into the liquid solvent. As the thermal diffusivity of gases is much smaller than that of liquids, it was assumed that heat was only transferred to and from the solvent phase on the wetted area of the packing, and so the heat balance over the gas phase was not modified. An extra ODE was also introduced to account for the temperature of the cooling fluid, $T_{\mathrm{Cw}}$ :

$$
\begin{equation*}
\frac{\mathrm{d} T_{\mathrm{CW}}}{\mathrm{dz}}=-\frac{1}{c_{p}^{\mathrm{CW}} L_{\mathrm{CW}}} a_{\mathrm{w}} q_{\mathrm{wall}} \tag{20}
\end{equation*}
$$


The temperature of the cooling water at the bottom of the column was also specified:

$$
\begin{equation*}
\left.T_{\mathrm{CW}}\right|_{z=0}=T_{\mathrm{CW}}^{\mathrm{in}} \tag{21}
\end{equation*}
$$


All other equations within the adiabatic model were retained.
The rate of heat transfer through the wall of the packing was calculated via the following equation:

$$
\begin{equation*}
q_{\mathrm{wall}}=h_{\mathrm{wall}}\left(T_{\mathrm{CW}}-T_{\mathrm{L}}\right) \xi(z) \tag{22}
\end{equation*}
$$

where $h_{\text {wall }}$ is the overall heat transfer coefficient, and $\xi(z)$ is a function which allows for the simulation of absorbers in which only part of the column contains a heat exchange packing. In regions of the column in which $\xi(z)=0$, there is no transfer of heat through the packing walls, and so $d T_{\mathrm{Cw}} / d z=0$ and the column acts adiabatically. In regions of the column in which $\xi(z)=1$, the solvent and cooling fluid are able to exchange heat. In this way, columns with multiple discrete regions of heat exchange packing may be simulated.

The performance of the heat exchange packing is strongly dependent on the value of the overall heat transfer coefficient, $h_{\text {wall }}$. Typical overall heat transfer coefficients in industrial plate heat exchangers range from 150 to $15,000 \mathrm{~W} / \mathrm{m}^{2} \mathrm{~K} .^{59}$ Miramontes et al. have reported values ranging from 35 to $147.9 \mathrm{~W} / \mathrm{m}^{2} \mathrm{~K}$ for the heat exchange packings they have tested. ${ }^{29,28}$ However, these heat exchange coefficients are in reference to the total geometric surface area of the packing, while $h_{\text {wall }}$ is in reference to the total wetted area, which is typically about $60 \%$ of the total area. ${ }^{48}$ Hence, the experiments of Miramontes et al. suggest values of $h_{\text {wall }}$ on the order of $50-250 \mathrm{~W} / \mathrm{m}^{2} \mathrm{~K}$.

Values of $h_{\text {wall }}$ were also estimated using an idealized Multiphysics model developed in COMSOL, in which flow of a 1 mm thick layer of fluid past a wall was simulated (see Figure S1). The flow of a cooling fluid on the other side of the wall was also simulated, and heat
transfer between the thin film, wall and cooling fluid was modeled, assuming the fluids had the thermophysical properties of water. The simulation predicted that the overall heat transfer coefficient would be on the order of $500 \mathrm{~W} / \mathrm{m}^{2} \mathrm{~K}$. This is slightly larger than the values measured experimentally by Miramotes et al., ${ }^{29}$ but is on the same order of magnitude. In most of the simulations that follow, we use relatively modest values of $h_{\text {wall }}$ ranging from 100 to $250 \mathrm{~W} / \mathrm{m}^{2} \mathrm{~K}$, consistent with available experimental data.

\section*{2.6 | Optimal temperature profiles}

The adiabatic model was modified to allow the numerical calculation of the optimal temperature profile along the length of the absorption column, defined as the temperature profile which minimized the column length required to achieve a specified degree of $\mathrm{CO}_{2}$ capture (e.g., $90 \%$ capture) for a given set of inlet flow rates. In the modified model, the liquid and gas temperatures were not calculated via energy balances, Equations (7) and (8). Instead, a single column temperature profile was specified by the user, and the mass balance equations, Equations (1-6), were then solved assuming that the gas and liquid temperatures conformed to this arbitrary profile. For the optimization calculations, the temperature profile was specified by providing a vector of 30 temperature values distributed evenly along the column length, and a smooth temperature profile was created from these values by interpolation with a cubic spline. The default differential evolution optimization algorithm from BlackBoxOptim.jl ${ }^{60}$ was then used to find the optimal temperature profile, which minimized the column height required to achieve a specified degree of $\mathrm{CO}_{2}$ capture for the given liquid and gas inlet conditions. The resulting optimal solution was found to be independent of the (randomly generated) starting population, and the same profile was found when the temperature profile was described via a 50 -point spline. As the optimized temperature profiles never resulted in strong mass transfer pinches, a single-point shooting method was found to be sufficient for these calculations. As this model converged in $\sim 1 \mathrm{~ms}$ on a single computational core (i7-8850H CPU 2.60 GHz ), its use significantly sped up the optimization calculations (which typically required $\sim 25,000$ independent simulations to converge).

\section*{3 | MODEL VALIDATION AND BEHAVIOR}

\section*{3.1 | Model validation}

The adiabatic model described above was validated against the pilot column data of Kvamsdal and Rochelle ${ }^{21}$ and Dugas et al., ${ }^{61}$ who both measured temperature profiles along a packed column in which $\mathrm{CO}_{2}$ was absorbed by aqueous MEA solutions. Validation against pilot column data may be approached in two ways: fix the $\mathrm{CO}_{2}$ capture fraction, and calculate the column height, or fix the column height and calculate the $\mathrm{CO}_{2}$ capture fraction; we discuss both approaches using the available data.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/7251841f-ade3-439f-839f-850b9c70e3cb-07.jpg?height=680&width=810&top_left_y=163&top_left_x=171}
\captionsetup{labelformat=empty}
\caption{FIGURE 4 Validation of absorption column model against (A) Case 32 and (B) Case 43 of the pilot column data reported by Kvamsdal and Rochelle. ${ }^{21}$ Inlet liquid and gas flowrates and temperatures taken from Table 2 of Kvamsdal and Rochelle ${ }^{21}$. Negative heights refer to liquid temperatures measured beneath the gas inlet}
\end{figure}

Kvamsdal and Rochelle reported temperature profile data for two cases (their "Case 32" and "Case 43"), and these cases were simulated using the adjusted inlet liquid and gas flow rates and reported fraction of $\mathrm{CO}_{2}$ captured found in tables 2 and 3 of their article. As shown in Figure 4, the predicted temperature profiles are in excellent agreement with the column data, with the discrepancy comparable to the gPROMS ${ }^{\circledR}$ model and Aspen Plus ${ }^{\circledR}$ model shown in Figures 2 and 3 of Kvamsdal et al. ${ }^{21}$ The predicted column heights are slightly shorter than the actual height: the model predicted 5.1 m and 4.5 m for Case 32 and 43 respectively, while the actual height was 6.1 m . However, such discrepancies are not unreasonable when using a general purpose correlation for $a_{\mathrm{w}} / a$ and $k_{\mathrm{G}}$. ${ }^{62}$ Dugas et al. ${ }^{61}$ reported temperature profile data for a 17 m tall column. In Figure S6, simulated temperature profiles are compared with their data at a wide range of flow rates and lean solvent loadings. In these simulations, the column height is fixed at 17 m . In each case, the fraction of $\mathrm{CO}_{2}$ captured and resulting temperature profiles predicted by the model are in excellent agreement with the experimental data. Overall, these results suggest that the first-principles model developed above is capable of accurately simulating the behavior of absorbers utilizing aqueous MEA.

\section*{3.2 | Behavior of the temperature bulge}

The temperature bulge is caused by the heat released from the exothermic absorption and condensation reactions occurring within the absorber. The temperature bulge may occur at the top, middle, or bottom of the column, ${ }^{63}$ and the primary factor determining its location is the $L / G$ ratio. As discussed by Kvamsdal and Rochelle, ${ }^{21}$ at large liquid flow rates the heat released within the column is principally

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/7251841f-ade3-439f-839f-850b9c70e3cb-07.jpg?height=1260&width=823&top_left_y=163&top_left_x=1071}
\captionsetup{labelformat=empty}
\caption{FIGURE 5 Behavior of temperature bulge. (A), (B) Temperature profiles for absorption columns operating at (A) $1.2 L_{\min }$ and (B) $2 L_{\min }$. See Table 2 for conditions. (C) Variation in column height, maximum magnitude of temperature bulge, and location of the maxima of the temperature bulge as a function of the liquid flow rate}
\end{figure}
carried out of the column via the liquid solvent, resulting in a small bulge near the bottom of the column. On the other hand, at large gas flow rates the excess enthalpy is carried out via the gas phase, resulting in a larger bulge (due to the smaller heat capacity of the gas) at the top of the column. The $L / G$ ratio at which the bulge transitions from the top to the bottom of the column may be determined via a total energy balance developed by Kvamsdal and Rochelle, ${ }^{21}$ who also demonstrated that the transition occurs quite abruptly in the vicinity of the critical liquid flowrate.

This qualitative description of the temperature bulge's behavior is consistent with the predictions of the model. In Figures 5(A), (B), temperature profiles are shown for an absorption column running under industrially relevant conditions at various liquid flow rates (see Table 2; identical conditions are used in a number of simulations which follow.) The minimum liquid flow rate was calculated by reducing the liquid flow rate to the point at which a $1 \%$ increase in $L$ would cause at least a $\sim 25 \%$ decrease in column height; at these flow rates the column was severely pinched (see Figure S4). In Figure 5(A), which has a relatively low liquid flow rate, the bulge is large and near the top

\begin{table}
\captionsetup{labelformat=empty}
\caption{TABLE 2 Process conditions}
\begin{tabular}{|l|l|}
\hline Parameter & Value \\
\hline Inlet gas flow rate, & $50 \mathrm{~mol} / \mathrm{m}^{2} \mathrm{~s}$ \\
\hline Inlet $\mathrm{CO}_{2}$ mole fraction & 0.1 \\
\hline Inlet $\mathrm{H}_{2} \mathrm{O}$ mole fraction & 0.05 \\
\hline Inlet gas temperature & 313.15 K \\
\hline Inlet MEA weight fraction & 0.3 \\
\hline Inlet (lean) liquid loading & 0.25 \\
\hline Inlet liquid temperature & 313.5 K \\
\hline Fraction of $\mathrm{CO}_{2}$ captured & 0.9 \\
\hline Total packing surface area & $150 \mathrm{~m}^{2} / \mathrm{m}^{3}$ \\
\hline Nominal packing size & 0.0253 m \\
\hline Void fraction & 0.97 \\
\hline
\end{tabular}
\end{table}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/7251841f-ade3-439f-839f-850b9c70e3cb-08.jpg?height=668&width=801&top_left_y=891&top_left_x=178}
\captionsetup{labelformat=empty}
\caption{FIGURE 6 Optimal and adiabatic temperature profiles under conditions shown in Table 2 for (A) $1.2 L_{\text {min }}$, (B) $1.5 L_{\text {min }}$, and (C) $2 L_{\text {min }}$}
\end{figure}
of the column, while in Figure 5(B), with a faster liquid flow rate, the magnitude of the bulge is smaller and it is near the bottom of the column. In Figure 5(C), the column height, bulge location and maximum bulge temperature are plotted as a function of the liquid flow rate. As predicted by Kvamsdal and Rochelle, ${ }^{21}$ at a critical liquid flowrate (around $1.9 L_{\text {min }}$ in this system) the bulge quickly reduces in magnitude and jumps from the top to the bottom of the column. The column height also decreases, as the very large temperature of the bulge near the top of the column is not optimal for $\mathrm{CO}_{2}$ absorption. The physical origin of the local maxima of the magnitude of the temperature bulge near $1.9 L_{\text {min }}$ is unclear, though identical behavior may be seen in Figure 6 of Kvamsdal et al. ${ }^{21}$

\section*{4 | OPTIMAL TEMPERATURE PROFILE}

In this section a number of optimal temperature profiles are calculated, which minimize the column height required to achieve a
fixed degree of separation for a given set of gas and liquid inlet flow rates. While these optimal profiles may not be realized in an actual system, their calculation provides a base case which is useful when comparing different intercooling approaches and determining whether more advanced intercooling may be worthwhile. The optimal temperature at any point along the column represents a trade-off between lowering the temperature to increase the absorption driving force and increasing the temperature to increase absorption kinetics.

Several optimal temperature profiles were calculated for the conditions shown in Table 2, which involved $90 \% \mathrm{CO}_{2}$ capture from a $10 \% \mathrm{CO}_{2}$ stream. As shown in Figures 6(A)-(C), for this system the optimal temperature profiles were cooler and more uniform than the adiabatic profiles. At lower liquid flow rates, for which the temperature bulge was larger and near the top of the column, the optimal profile led to a column about $20 \%$ shorter than the adiabatic case (see Figures 6(A), (B)). In Figure 6(C), the liquid flow rate was large enough to reduce the magnitude of the temperature bulge and shift it to the bottom of the column, and in this case the height of the adiabatic column was only marginally greater than the minimum possible height. These results suggest that intercooling is most valuable for systems in which the temperature bulge is large and near the top of the column.

Optimal profiles were also calculated for absorption columns treating a flue gas stream from a natural gas plant which contained $5 \% \mathrm{CO}_{2}$, and for a more concentrated stream containing $20 \% \mathrm{CO}_{2}$ (see Figure S5). For the $5 \% \mathrm{CO}_{2}$ case, the optimal profile was only $6 \%$ smaller than the adiabatic case, while for the flue gas containing $20 \% \mathrm{CO}_{2}$, the optimal profile was almost $40 \%$ smaller than the adiabatic case. This trend is unsurprising, as the temperature bulge is much larger when treating more concentrated flue gas streams. For these systems, it is likely that some form of intercooling is essential.

\section*{5 | HEAT INTEGRATION VIA HEAT EXCHANGE PACKINGS}

In this section, a number of novel absorption processes utilizing heat exchange packings are modeled and evaluated. These processes are enabled by the excellent heat transfer properties of these structures, and may be used to reduce the column height, reduce cooling duties, and enable waste heat from the absorber to be efficiently recovered.

\section*{5.1 | Traditional intercooling using heat exchange packings}

A straightforward application of heat exchange packings is shown in Figure 7(A), in which a cooling fluid is used to cool the solvent and gas streams using heat exchange packings placed along a section of the column. This arrangement is similar to traditional in-and-out intercooling (see Figure 1), but eliminates the external cooling unit and the need for solvent redistribution within the column.
![](https://cdn.mathpix.com/cropped/7251841f-ade3-439f-839f-850b9c70e3cb-09.jpg?height=688&width=510&top_left_y=174&top_left_x=352)
![](https://cdn.mathpix.com/cropped/7251841f-ade3-439f-839f-850b9c70e3cb-09.jpg?height=699&width=805&top_left_y=174&top_left_x=904)

FIGURE 7 Traditional intercooling via heat exchange packings. (A) Diagram of an absorber utilizing heat exchange packings. (B), (C) Temperature profile of cooled liquid solvent, cooling fluid, and adiabatic solvent under conditions shown in Table 2 at $1.2 L_{\text {min }}$. Cooling water flowrate: $500 \mathrm{~mol} / \mathrm{m}^{2} \mathrm{~s}$, where " $\mathrm{m}^{2}$ " refers to cross-sectional area of column. This flowrate corresponds to a superficial velocity of $0.9 \mathrm{~cm} / \mathrm{s}$. Minimum possible column height taken from Figure 6. (B) $15 \%$ of column utilizes heat exchange packing with $h=250 \mathrm{~W} / \mathrm{m}^{2} \mathrm{~K}$ (C) $25 \%$ of column utilizes heat exchange packing with $h=100 \mathrm{~W} / \mathrm{m}^{2} \mathrm{~K}$

In order to investigate this approach, a number of simulations were conducted using the process conditions described in Table 2. Simulations were conducted assuming overall heat transfer coefficients of 100 and $250 \mathrm{~W} / \mathrm{m}^{2} \mathrm{~K}$. In each case the position of the heat exchange packing was optimized to minimize the column height. As may be seen in Figure 7(B), for $h=250 \mathrm{~W} / \mathrm{m}^{2} \mathrm{~K}$, intercooling along only $15 \%$ of the column could reduce the column height by $13 \%$. For $h=100 \mathrm{~W} / \mathrm{m}^{2} \mathrm{~K}$, comparable improvements could be achieved when $25 \%$ of the column was used (Figure 7(C)).

The superficial velocity of the cooling fluid was set to $0.9 \mathrm{~cm} / \mathrm{s}$. This is similar to the solvent superficial velocity. Increasing the cooling water flow rate beyond this value did not significantly improve process performance. However, if the cooling water flow rate is reduced below that of the solvent, the relatively small thermal mass of the cooling water begins to harm process performance. In Section S3 of the Supporting Information, the commercial software COMSOL is used to calculate the pressure drop within the cooling water in a TPMS structure. The resulting pressure drops are modest $(\sim 1 \mathrm{kPa} / \mathrm{m}$ column) and will not significantly increase utility pumping duties.

It is interesting to note that a simple intercooling arrangement is able to reduce the height of the column to close to the minimum possible value. As traditional in-and-out intercooling is likely to result in similar temperature profiles, achieving near-optimal performance in traditional intercooling should also be possible. This result has not previously been demonstrated within the literature, as researchers have not previously sought to calculate optimal temperature profiles within absorbers for carbon capture. It suggests that alternative intercooling schemes which aim to be more effective than traditional intercooling (such as the "advanced" recycle intercooling discussed by Rezazadeh et al. ${ }^{23}$ ) may lead to only marginal improvements in column performance.

\section*{5.2 | Waste heat recovery via cool rich stream}

Another simple yet promising approach to utilizing heat exchange packings is the use of the cool, rich stream as the cooling fluid (see Figure 8(A)). In this arrangement, heat generated by the exothermic absorption reactions is transferred to the cool, rich stream, preheating it before it is sent to the cross heat exchanger. Alternative arrangements in which only a fraction of the cool, rich solvent is used for cooling could also be devised. A similar concept was proposed by Geleff ${ }^{64}$ using traditional, external cooling, and the approach has several advantages over traditional intercooling.

First, by using the cool rich solvent as the cooling fluid, the cooling load associated with the external cooling unit in a traditional in-and-out intercooler may be reduced or eliminated. This cooling load is not insignificant: in a CCS system with traditional in-and-out cooling, typically about $1-2 \mathrm{GJ}$ of heat is removed per $\mathrm{tCO}_{2}$ captured by the absorber. This is comparable to the reboiler heat duty, ${ }^{15}$ though of course the equivalent work required to remove this lowgrade heat using a cooler is much smaller the equivalent work required to supply heat in the stripper.

Second, this approach allows for the recovery of waste heat from the exothermic reactions occurring within the absorber. These reactions are the reverse of the endothermic reactions occurring in the stripper, which are responsible for a significant fraction of the reboiler heat duty. ${ }^{65}$ Recovery of this waste heat could reduce exergetic losses within the absorber, which account for approximately a quarter of all exergetic losses within a modern carbon capture process. ${ }^{15}$ Of course, the recovery of heat from a particular unit does not necessary imply that the entire process can be run more efficiently, especially when the recovered heat is of low quality. However, it is possible that, in this particular case, the presence of a cool side pinch in the main heat

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/7251841f-ade3-439f-839f-850b9c70e3cb-10.jpg?height=655&width=1313&top_left_y=169&top_left_x=379}
\captionsetup{labelformat=empty}
\caption{FIGURE 8 Intercooling via cool rich stream using heat exchange packings (A) Diagram of absorber utilizing cool rich stream within heat exchange packings (B), (C) Temperature profiles for absorber cooled by redirecting cool, rich solvent through heat exchange packings within column. Profiles shown for liquid solvent within column, solvent used as cooling fluid, and solvent within an adiabatic column. Minimum possible column height also shown (cf., Figure 6.) Conditions taken from Table 2 at $1.2 L_{\min }$, with the exception of $\left.T_{L}\right|_{z=\mathscr{L}}$, which is set to $60^{\circ} \mathrm{C}$. (B) $h=250 \mathrm{~W} / \mathrm{m}^{2} \mathrm{~K}$, heat exchange packing in along $15 \%$ of column. (C) $h=100 \mathrm{~W} / \mathrm{m}^{2} \mathrm{~K}$, heat exchange packing along $30 \%$ of column}
\end{figure}
exchanger may allow for a modest reduction in the reboiler duty, on the order of 1\% (see Supporting Information, Section S2).

In Figures 8(B), (C), simulations of this approach are shown. The process conditions were taken from Table 2, with the exception of the lean solvent temperature, which was increased to $60^{\circ} \mathrm{C}$ so that it would be hotter than the preheated cool, rich stream. As the solvent was used as the cooling fluid, the cooling fluid's thermophysical properties were set equal to those of the solvent, and the inlet temperature and flow rate of the cooling fluid were set equal to the outlet flow rate and temperature of the solvent. The endothermic evolution of $\mathrm{CO}_{2}$ as the cool rich solvent is warmed was not modeled.

In Figure 8(B), a section of heat transfer packing taking up $15 \%$ of the column is used, with overall heat transfer coefficient $h=250 \mathrm{~W} / \mathrm{m}^{2} \mathrm{~K}$. In Figure 8(C), a section spanning $30 \%$ of the column is used, and $h$ takes a lower value of $100 \mathrm{~W} / \mathrm{m}^{2} \mathrm{~K}$. In each case, intercooling is able to reduce the column height by $\sim 10 \%$, while preheating the cool, rich solvent by $8-12^{\circ} \mathrm{C}$. It is clear that the rate of heat transfer and the thermal mass of the cool, rich stream are both sufficient to facilitate effective intercooling. However, the effect of preheating the cool, rich stream, and increasing the temperature of the cool, lean solvent are less clear. Overall, this intercooling approach shows some promise, though further process-scale simulation is required for a thorough evaluation.

\section*{5.3 | Adiabatic heat redistribution within the column}

As discussed above, the optimal temperature profile tends to be relatively uniform and at an intermediate temperature. This suggests the possibility of "flattening out" the temperature profile by redistributing heat from warmer parts of the column to cooler regions, with no net addition or removal of heat to the column. Such an approach would
eliminate the cooling load associated with intercooling, and, given the small temperature differences involved, it is well suited to heat exchange packings.

Consider the process flow diagram shown in Figure 9(A). By running the cooling fluid through a closed cycle, it is possible to redistribute heat from warmer to cooler regions with no net removal of heat. Two simulations of this approach are shown in Figures 9(B), (C), each using a relatively large heat transfer coefficient of $h=1000 \mathrm{~W} / \mathrm{m}^{2} \mathrm{~K}$. Notice that in each figure, the temperature of the exiting cooling fluid (near the top of the column) equals the temperature of the entering fluid (near the bottom of the column), and so the cooling fluid can be circulated in a continuous, closed loop. These profiles were calculated using an optimization algorithm which used a severe penalty function to minimize the discrepancy between the inlet and outlet cooling fluid temperatures, while also minimizing the column height. In particular, the optimization algorithm sought to minimize:

$$
\begin{equation*}
f(x)=1000\left|T_{C W}^{\mathrm{top}}-T_{C W}^{\mathrm{bot}}\right|+\mathscr{L} \tag{23}
\end{equation*}
$$

where $\mathscr{L}$ is the column length and $T_{C W}^{\text {top }}$ and $T_{C W}^{\text {bot }}$ are the temperature of the cooling water at the top and bottom of the column. Both the inlet cooling water temperature and the position of the sections of heat exchange packing were independent variables within the optimization. In Figure 9(B), the column height was reduced by $8 \%$ using heat exchange packings on only $10 \%$ of the column height. If heat exchange packings are used throughout the column (Figure 9(C)), a near-optimal, uniform temperature profile may be achieved.

One advantage of this approach is that it does not rely on heat integration with other parts of the capture process, and could be applied in situations where the approaches discussed above are impossible or unhelpful. Elimination of the external cooling unit and the cooling load could also reduce capital cost and energy use.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/7251841f-ade3-439f-839f-850b9c70e3cb-11.jpg?height=746&width=1413&top_left_y=161&top_left_x=324}
\captionsetup{labelformat=empty}
\caption{FIGURE 9 Closed cooling water cycle, in which heat is redistributed within column using heat exchange packings. (A) Diagram of closed intercooling cycle. (B), (C) Temperature profiles for solvent cooled via closed cooling cycle. Profiles shown for liquid solvent within column, cooling fluid, and solvent within an adiabatic column. Conditions taken from Table 2 at $1.2 L_{\text {min }}$. Cooling water flow rate: $1000 \mathrm{~mol} / \mathrm{m}^{2} \mathrm{~s}$, where " $\mathrm{m}^{2 \text { " }}$ refers to the cross-sectional area of the column. Minimum possible column height also shown (cf., Figure 6; B) Two sections of heat exchange packing along a total of $10 \%$ of the column, $h=1000 \mathrm{~W} / \mathrm{m}^{2} \mathrm{~K}$. (C) Active packing along entire column, $h=1000 \mathrm{~W} / \mathrm{m}^{2} \mathrm{~K}$}
\end{figure}

Unfortunately, when the heat transfer coefficient was reduced to $h=500$ or $100 \mathrm{~W} / \mathrm{m}^{2} \mathrm{~K}$, no configurations were found in which a closed cooling fluid loop could be established within the column. The closure of the cooling cycle in Figures 9(B), (C) depended upon the very fast cooling of the cooling water in the narrow, cool region above the temperature bulge, and when the heat transfer coefficient was reduced, insufficient heat transfer occurred in this region to close the cooling water cycle. This suggests that these configurations will only be possible if heat exchange packings with superior heat transfer coefficients can be designed. It is possible that the loop could be closed by transferring some heat from the cooling water to the cool, lean solvent prior to the solvent entering the absorber. However, this would require the installation of an additional heat exchange unit.

\section*{6 | DISCUSSION AND CONCLUSIONS}

The simulation results presented above demonstrate that even small sections of heat exchange packing are capable of removing and recovering industrially-relevant quantities of heat. A number of novel process configurations were investigated, and in each case heat exchange packings were able to reduce the column height by $\sim 10 \%$ relative to the adiabatic case (though for adiabatic heat redistribution, a large heat transfer coefficient was required.) In and of itself this is a relatively modest improvement, comparable to that provided by traditional intercooling. However, heat exchange packings may provide several auxiliary benefits beyond the reduction in column height, including the elimination of external cooling units, reduction or elimination of cooling duty, and recovery of waste heat from the absorber. Intercooling via the cool rich stream and adiabatic heat redistribution
are particularly well suited to heat exchange packings, as they require large quantities of heat transfer between streams of similar temperature. It is likely that if heat exchange packings are to be applied industrially, it will be within the context of a broader heat integration strategy, rather than simply as a replacement for an in-and-out intercooling unit. For this reason, future studies should include processscale simulations, in order to accurately quantify the benefits these materials can provide.

While this study has focused on $\mathrm{CO}_{2}$ absorption into MEA, it is likely that heat exchange packings will be even more effective when applied to third generation, water lean CCS solvents such as CO2BOLs. The low heat capacity of these solvents, and the fact that they tend to absorb $\mathrm{CO}_{2}$ faster at lower temperatures, will likely render some form of intercooling essential. ${ }^{17}$

Heat exchange packings may also be applied to other process operations. For example, it is possible that direct addition of heat within the packing of a stripper column could be used to improve column performance. Such a design could be particularly useful for water-lean solvents which cannot be directly contacted with large quantities of steam. Heat exchange packings could also be used to expand the design space available for distillation processes, which are responsible for $90-95 \%$ of industrial liquid separations, and which typically operate well below the thermodynamic optimum. ${ }^{66}$ Applications to solvent extraction columns and multiphase chemical reactors are also possible.

As the design space for 3D-printed packings is effectively unlimited, there is significant scope for improvements in the performance of the packings themselves, primarily measured in terms of mass transfer and heat transfer rates and pressure drop. For example, TPMS structures have been shown to provide excellent transport
properties for single-phase systems, ${ }^{31}$ and it is possible (though far from certain) that they may provide similar benefits when used in multiphase, reactive systems. It is also possible that topological optimization techniques could be used to inform packing design ${ }^{67}$ though this is a difficult challenge for reactive two phase systems. Further work is also required to assess hydrodynamic properties of TPMS packings, including pressure drop and flooding point. If flooding occurs at relatively low gas flow rates in heat exchange packings, the absorber cross-sectional area and capital cost will increase.

The design and large-scale manufacture of these materials is strongly dependent on the ongoing development of additive manufacturing technologies. At present, additive manufacturing of metal parts remains expensive, but cost reductions are expected over the coming decade, ${ }^{68,69}$ largely driven by increased use of additive manufacturing in the aerospace, healthcare and energy sectors. ${ }^{70}$ Metal additive manufacturing is presently being considered for the design of structured catalysts, mixers and chemical reactors. ${ }^{71,72}$ If these materials can provide significant improvements in process performance, it is plausible that they will be cost effective in the near future. Nevertheless, at present capital cost is likely the most significant barrier to the large-scale implementation of these technologies.

Overall, heat exchange packings are a promising means of intensifying heat removal and recovery in absorption columns. They may be used to reduce the height of an absorber to close to the minimum possible value, to improve absorber performance by adiabatically redistributing heat within the column, and to recover waste heat from the exothermic reactions within the absorber. Future work should focus on improved packing design, applications to other industrial processes, and the rigorous simulation of heat integration strategies on a process-wide scale.

\section*{ACKNOWLEDGMENTS}

This work was performed under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344. Funding was provided by the U.S. Department of Energy, Office of Fossil Energy under FWP-FEW0225. Release Number: LLNL-JRNL-814462.

\section*{CONFLICT OF INTEREST}

The authors declare no potential conflict of interest.

\section*{DATA AVAILABILITY STATEMENT}

The data that support the findings of this study are available from the corresponding author upon reasonable request.

\section*{ORCID}

Thomas Moore (D) https://orcid.org/0000-0003-0802-5547

\section*{REFERENCES}
1. Stocker TF, Qin D, Plattner GK, et al. Climate change 2013: the physical science basis. Contribution of Working Group I to the Fifth Assessment Report of the Intergovernmental Panel on Climate Change; 2013; 1535.
2. Wilcox J. Carbon Capture. New York: Springer Science \& Business Media; 2012.
3. Baena-Moreno FM, Rodríguez-Galán M, Vega F, Alonso-Fariñas B, Vilches Arenas LF, Navarrete B. Carbon capture and utilization technologies: a literature review and recent advances. Energy Sources, Part A: Recovery, Utilization, and Environmental Effects. 2019;41(12):1403-1433.
4. Bui M, Adjiman CS, Bardow A, et al. Carbon capture and storage (CCS): the way forward. Energ Environ Sci. 2018;11(5):1062-1176.
5. Mumford KA, Wu Y, Smith KH, Stevens GW. Review of solvent based carbon-dioxide capture technologies. Front Chem Sci Eng. 2015;9(2): 125-141.
6. Sjostrom S, Krutka H. Evaluation of solid sorbents as a retrofit technology for CO2 capture. Fuel. 2010;89(6):1298-1306.
7. Khalilpour R, Mumford K, Zhai H, Abbas A, Stevens G, Rubin ES. Membrane-based carbon capture from flue gas: a review. J Clean Prod. 2015;103:286-300.
8. Moore T, Biviano M, Mumford KA, Dagastine RR, Stevens GW, Webley PA. Solvent impregnated polymers for carbon capture. Ind Eng Chem Res. 2019;58(16):6626-6634.
9. Vericella JJ, Baker SE, Stolaroff JK, et al. Encapsulated liquid sorbents for carbon dioxide capture. Nat Commun. 2015;6(1):1-7.
10. Nguyen D, Murialdo M, Hornbostel K, et al. 3D printed polymer composites for CO2 capture. Ind Eng Chem Res. 2019;58(48):2201522020.
11. Rochelle GT. Amine scrubbing for CO2 capture. Science. 2009;325 (5948):1652-1654.
12. Institute GC. Institute. The global status of CCS: 2017. Canberra, Australia 2017;.
13. NETL, Performance baseline for fossil energy plants, Volume 1: Bituminous coal and natural gas to electricity, Revision 2, November 2010. DOE/NETL-2010/1397; 2012.
14. Lin YJ, Rochelle GT. Approaching a reversible stripping process for CO2 capture. Chem Eng J. 2016;283:1033-1043.
15. Lin YJ, Chen E, Rochelle GT. Pilot plant test of the advanced flash stripper for CO2 capture. Faraday Discuss. 2016;192:37-58.
16. Jiang K, Li K, Yu H, Chen Z, Wardhaugh L, Feron P. Advancement of ammonia based post-combustion CO2 capture using the advanced flash stripper process. Appl Energy. 2017;202:496-506.
17. Heldebrant DJ, Koech PK, Glezakou VA, Rousseau R, Malhotra D, Cantu DC. Water-lean solvents for post-combustion CO2 capture: fundamentals, uncertainties, opportunities, and outlook. Chem Rev. 2017;117(14):9594-9624.
18. Jiang Y, Mathias PM, Whyatt G, Freeman C, Zheng F, Glezakou VA, et al. Attempting to break the $2 \mathrm{GJ} /$ tonne CO 2 barrier; Development of an advanced water-lean capture solvent from molecules to detailed process design. Development of an advanced water-lean capture solvent from molecules to detailed process design (April 29, 2019) 2019;Available at SSRH https://ssrn.com/abstract= 3379731.
19. Yuan Y, Rochelle GT. Lost work: a comparison of water-lean solvent to a second generation aqueous amine process for CO 2 capture. Int $J$ Greenhouse Gas Control. 2019;84:82-90.
20. Le Moullec Y, Neveux T, Al Azki A, Chikukwa A, Hoff KA. Process modifications for solvent-based post-combustion CO2 capture. Int J Greenhouse Gas Control. 2014;31:96-112.
21. Kvamsdal HM, Rochelle GT. Effects of the temperature bulge in CO2 absorption from flue gas by aqueous monoethanolamine. Ind Eng Chem Res. 2008;47(3):867-875.
22. Walters MS, Edgar TF, Rochelle GT. Dynamic modeling and control of an intercooled absorber for post-combustion CO2 capture. Chem Eng Process-Process Intens. 2016;107:1-10.
23. Rezazadeh F, Gale WF, Rochelle GT, Sachde D. Effectiveness of absorber intercooling for CO 2 absorption from natural gas fired flue gases using monoethanolamine solvent. Int J Greenhouse Gas Control. 2017;58:246-255.
24. Freguia S, Rochelle GT. Modeling of CO2 capture by aqueous monoethanolamine. AIChE J. 2003;49(7):1676-1686.
25. Sachde D, Rochelle GT. Absorber intercooling configurations using aqueous piperazine for capture from sources with 4 to $27 \% \mathrm{CO} 2$. Energy Procedia. 2014;63:1637-1656.
26. Plaza JM, Chen E, Rochelle GT. Absorber intercooling in CO2 absorption by piperazine-promoted potassium carbonate. AIChE J. 2010;56 (4):905-914.
27. Oko E, Ramshaw C, Wang M. Study of intercooling for rotating packed bed absorbers in intensified solvent-based CO2 capture process. Appl Energy. 2018;223:302-316.
28. Miramontes E, Love L, Lai C, Sun X, Tsouris C. Additively manufactured packed bed device for process intensification of CO2 absorption and other chemical processes. Chem Eng J. 2020;388: 124092.
29. Miramontes E, Jiang EA, Love LJ, Lai C, Sun X, Tsouris C. Process intensification of CO 2 absorption using a 3D printed intensified packing device. AIChE J. 2019;66(8):e16285.
30. Chandrasekaran G. 3D printed heat exchangers an experimental study. Mechanical Engineering 2018.
31. Femmer T, Kuehne AJ, Wessling M. Estimation of the structure dependent performance of 3-D rapid prototyped membranes. Chem Eng J. 2015;273:438-445.
32. Saimpert M, Puxty G, Qureshi S, Wardhaugh L, Cousins A. A new rate based absorber and desorber modelling tool. Chem Eng Sci. 2013;96: 10-25.
33. Pandya J. Adiabatic gas absorption and stripping with chemical reaction in packed towers. Chem Eng Commun. 1983;19(4-6):343-361.
34. Tontiwachwuthikul P, Meisen A, Lim CJ. CO2 absorption by NaOH, monoethanolamine and 2-amino-2-methyl-1-propanol solutions in a packed column. Chem Eng Sci. 1992;47(2):381-390.
35. Simon LL, Elias Y, Puxty G, Artanto Y, Hungerbuhler K. Rate based modeling and validation of a carbon-dioxide pilot plant absorbtion column operating on monoethanolamine. Chem Eng Res Design. 2011; 89(9):1684-1692.
36. Kenig EY, Kucka L, Górak A. Rigorous modeling of reactive absorption processes. Chem Eng Technol: Ind Chem-Plant Equipment-Process EngBiotechnol. 2003;26(6):631-646.
37. Kucka L, Müller I, Kenig EY, Górak A. On the modelling and simulation of sour gas absorption by aqueous amine solutions. Chem Eng Sci. 2003;58(16):3571-3578.
38. Kvamsdal HM, Jakobsen JP, Hoff KA. Dynamic modeling and simulation of a CO2 absorber column for post-combustion CO2 capture. Chem Eng Process: Process Intensification. 2009;48(1):135-144.
39. Asprion N. Nonequilibrium rate-based simulation of reactive systems: simulation model, heat transfer, and influence of film discretization. Ind Eng Chem Res. 2006;45(6):2054-2069.
40. Lawal A, Wang M, Stephenson P, Koumpouras G, Yeung H. Dynamic modelling and analysis of post-combustion CO2 chemical absorption process for coal-fired power plants. Fuel. 2010;89(10):2791-2801.
41. Macías-Salinas R, Fair JR. Axial mixing in modern packings, gas, and liquid phases: II. Two-phase flow. AIChE J. 2000;46(1):79-91.
42. Weiland RH, Dingman JC, Cronin DB. Heat capacity of aqueous monoethanolamine, diethanolamine, N-methyldiethanolamine, and N-methyldiethanolamine-based blends with carbon dioxide. J Chem Eng Data. 1997;42(5):1004-1006.
43. Chiu LF, Li MH. Heat capacity of alkanolamine aqueous solutions. J Chem Eng Data. 1999;44(6):1396-1401.
44. Kim I, Svendsen HF. Heat of absorption of carbon dioxide (CO2) in monoethanolamine (MEA) and 2-(aminoethyl) ethanolamine (AEEA) solutions. Ind Eng Chem Res. 2007;46(17):5803-5809.
45. Mathias PM. The Gibbs-Helmholtz equation in chemical process technology. Ind Eng Chem Res. 2016;55(4):1076-1087.
46. Zakeri A, Einbu A, Svendsen HF. Experimental investigation of pressure drop in structured packings. Chem Eng Sci. 2012;73:285-298.
47. Sebastia-Saez D, Gu S, Ranganathan P, Papadikis K. Meso-scale CFD study of the pressure drop, liquid hold-up, interfacial area and mass transfer in structured packing materials. Int J Greenhouse Gas Control. 2015;42:388-399.
48. Onda K, Takeuchi H, Okumoto Y. Mass transfer coefficients between gas and liquid phases in packed columns. J Chem Eng Japan. 1968;1 (1):56-62.
49. Danckwerts PV, Lannus A. Gas-liquid reactions. J Electrochem Soc. 1970;117(10):369C.
50. Gabrielsen J, Michelsen ML, Stenby EH, Kontogeorgis GM. A model for estimating CO2 solubility in aqueous alkanolamines. Ind Eng Chem Res. 2005;44(9):3348-3354.
51. Pintola T, Tontiwachwuthikul P, Meisen A. Simulation of pilot plant and industrial CO2-MEA absorbers. Gas Separation \& Purification. 1993;7(1):47-52.
52. Astarita G. Mass Transfer with Chemical Reaction. London: Elsevier; 1967.
53. Billet R, Schultes M. Predicting mass transfer in packed columns. Chem Eng Technol: Ind Chem-Plant Equipment-Process Eng-Biotechnol. 1993;16(1):1-9.
54. Denbigh KG, Denbigh KG. The Principles of Chemical Equilibrium: with Applications in Chemistry and Chemical Engineering. London: Cambridge University Press; 1981.
55. Bird RB, Stewart WE, Lightfoot EN. Transport Phenomena. New York: John Wiley \& Sons; 1960.
56. Morrison DD, Riley JD, Zancanaro JF. Multiple shooting method for two-point boundary value problems. Commun ACM. 1962;5(12): 613-614.
57. Bezanson J, Edelman A, Karpinski S, Shah VB. Julia: a fresh approach to numerical computing. SIAM Rev. 2017;59(1):65-98. https://doi. org/10.1137/141000671.
58. Rackauckas C, Nie Q. Differentialequations. Jl-a performant and feature-rich ecosystem for solving differential equations in julia. J Open Res Software. 2017;5(1):1-10.
59. Vlasogiannis P, Karagiannis G, Argyropoulos P, Bontozoglou V. Airwater two-phase flow and heat transfer in a plate heat exchanger. Int J Multiphase Flow. 2002;28(5):757-772.
60. Feldt R, BlackBoxOptim.jl. GitHub; 2018. https://github.com/ robertfeldt/BlackBoxOptim.jl.
61. Dugas R, Alix P, Lemaire E, Broutin P, Rochelle G. Absorber model for CO2 capture by monoethanolamine-application to CASTOR pilot results. Energy Procedia. 2009;1(1):103-107.
62. Hegely L, Roesler J, Alix P, Rouzineau D, Meyer M. Absorption methods for the determination of mass transfer parameters of packing internals: a literature review. AIChE J. 2017;63(8):3246-3275.
63. Yokoyama T. Japanese R\&D on large-scale CO2 capture. ECI Conference on Separations Technology VI: New Perspectives on Very Large-Scale Operations, Fraser Island, Australia 2004;.
64. Geleff S. Method for recovery of carbon dioxide from a gaseous source. Patent No WO 2004;73838:A1.
65. Oexmann J, Kather A. Minimising the regeneration heat duty of postcombustion CO2 capture by wet chemical absorption: the misguided focus on low heat of absorption solvents. Int J Greenhouse Gas Control. 2010;4(1):36-43.
66. Agrawal R, Gooty RT. Misconceptions about efficiency and maturity of distillation. AIChE J. 2020;66(8):e16294.
67. Bendsoe MP, Sigmund O. Topology Optimization: Theory, Methods, and Applications. New York: Springer Science \& Business Media; 2013.
68. Baumers M, Holweg M. On the economics of additive manufacturing: experimental findings. J Operations Manag. 2019;65(8):794-809.
69. Heinen JJ, Hoberg K. Assessing the potential of additive manufacturing for the provision of spare parts. J Operations Manag. 2019;65(8):810-826.
70. Debroy T, Mukherjee T, Milewski J, et al. Scientific, technological and economic issues in metal printing and their solutions. Nat Mater. 2019;18(10):1026-1032.
71. Parra-Cabrera C, Achille C, Kuhn S, Ameloot R. 3D printing in chemical engineering and catalytic technology: structured catalysts, mixers and reactors. Chem Soc Rev. 2018;47(1):209-230.
72. Manoharan S, Lee K, Freiberg L, Coblyn M, Jovanovic G, Paul BK. Comparing the economics of metal additive manufacturing processes for micro-scale plate reactors in the chemical process industry. Procedia Manufact. 2019;34:603-612.
73. Weiland RH, Dingman JC, Cronin DB, Browning GJ. Density and viscosity of some partially carbonated aqueous alkanolamine solutions and their blends. J Chem Eng Data. 1998;43(3):378-382.
74. Li MH, Lai MD. Solubility and diffusivity of N 2 O and CO 2 in (Monoethanolamine +N -Methyldiethanolamine + water) and in (Monoethanolamine+ 2-Amino-2-methyl-1-propanol+ water). J Chem Eng Data. 1995;40(2):486-492.
75. Ma'mun S, Svendsen HF. Solubility of N2O in aqueous monoethanolamine and 2-(2-Aminoethyl-amino) ethanol solutions from 298 to 343 K. Energy Procedia. 2009;1(1):837-843.
76. Versteeg G, Van Dijck L, van Swaaij WPM. On the kinetics between CO 2 and alkanolamines both in aqueous and nonaqueous solutions. An Overview. Chem Eng Commun. 1996;144(1): 113-158.
77. Linstrom PJ, Mallard WG. NIST Chemistry WebBook, NIST Standard Reference Database Number 69. National Institute of Standards and Technology; 2020 (accessed July 15, 2020).
78. Bolz RE. CRC Handbook of Tables for Applied Engineering Science. London: CRC Press; 1973.
79. Company C. Flow of fluids through valves, fittings and pipe. Technical paper No. 410 (TP 410); 1988.
80. Wilke C. A viscosity equation for gas mixtures. J Chem Phys. 1950;18 (4):517-519.
81. Kestin J, Sokolov M, Wakeham WA. Viscosity of liquid water in the range- 8 C to 150 C. J Phys Chem Ref Data Monogr. 1978;7(3):941-948.
82. Stull DR. Vapor pressure of pure substances. Organic and inorganic compounds. Ind Eng Chem. 1947;39(4):517-540.

\section*{SUPPORTING INFORMATION}

Additional supporting information may be found online in the Supporting Information section at the end of this article.

How to cite this article: Moore T, Nguyen D, Iyer J, Roy P, Stolaroff JK. Advanced absorber heat integration via heat exchange packings. AIChE J. 2021;67:e17243. https://doi.org/ $\underline{10.1002 / \text { aic. } 17243}$