"""Plot a CLF contour"""
from typing import cast, List, Tuple, Optional, TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import tqdm

from neural_clbf.experiments import Experiment

if TYPE_CHECKING:
    from neural_clbf.controllers import Controller, CLFController  # noqa


class CLFContourExperiment(Experiment):
    """An experiment for plotting the contours of learned CLFs"""

    def __init__(
        self,
        name: str,
        domain: Optional[List[Tuple[float, float]]] = None,
        n_grid: int = 50,
        x_axis_index: int = 0,
        y_axis_index: int = 1,
        x_axis_label: str = "$x$",
        y_axis_label: str = "$y$",
        default_state: Optional[torch.Tensor] = None,
        plot_unsafe_region: bool = True,
    ):
        """Initialize an experiment for plotting the value of the CLF over selected
        state dimensions.

        args:
            name: the name of this experiment
            domain: a list of two tuples specifying the plotting range,
                    one for each state dimension.
            n_grid: the number of points in each direction at which to compute V
            x_axis_index: the index of the state variable to plot on the x axis
            y_axis_index: the index of the state variable to plot on the y axis
            x_axis_label: the label for the x axis
            y_axis_label: the label for the y axis
            default_state: 1 x dynamics_model.n_dims tensor of default state
                           values. The values at x_axis_index and y_axis_index will be
                           overwritten by the grid values.
            plot_unsafe_region: True to plot the safe/unsafe region boundaries.
        """
        super(CLFContourExperiment, self).__init__(name)

        # Default to plotting over [-1, 1] in all directions
        if domain is None:
            domain = [(-1.0, 1.0), (-1.0, 1.0)]
        self.domain = domain

        self.n_grid = n_grid
        self.x_axis_index = x_axis_index
        self.y_axis_index = y_axis_index
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.default_state = default_state
        self.plot_unsafe_region = plot_unsafe_region

    @torch.no_grad()
    def run(self, controller_under_test: "Controller") -> pd.DataFrame:
        """
        Run the experiment, likely by evaluating the controller, but the experiment
        has freedom to call other functions of the controller as necessary (if these
        functions are not supported by all controllers, then experiments will be
        responsible for checking compatibility with the provided controller)

        args:
            controller_under_test: the controller with which to run the experiment
        returns:
            a pandas DataFrame containing the results of the experiment, in tidy data
            format (i.e. each row should correspond to a single observation from the
            experiment).
        """
        # Sanity check: can only be called on a NeuralCLFController
        if not (
            hasattr(controller_under_test, "V")
            and hasattr(controller_under_test, "solve_CLF_QP")
        ):
            raise ValueError("Controller under test must be a CLFController")

        controller_under_test = cast("CLFController", controller_under_test)

        # Set up a dataframe to store the results
        results = []

        # Set up the plotting grid
        device = "cpu"
        if hasattr(controller_under_test, "device"):
            device = controller_under_test.device  # type: ignore

        x_vals = torch.linspace(
            self.domain[0][0], self.domain[0][1], self.n_grid, device=device
        )
        y_vals = torch.linspace(
            self.domain[1][0], self.domain[1][1], self.n_grid, device=device
        )

        # Default to all zeros if no default provided
        if self.default_state is None:
            default_state = torch.zeros(1, controller_under_test.dynamics_model.n_dims)
        else:
            default_state = self.default_state

        default_state = default_state.type_as(x_vals)

        # Make a copy of the default state, which we'll modify on every loop
        x = (
            default_state.clone()
            .detach()
            .reshape(1, controller_under_test.dynamics_model.n_dims)
        )

        # Loop through the grid
        prog_bar_range = tqdm.trange(self.n_grid, desc="Plotting CLF", leave=True)
        for i in prog_bar_range:
            for j in range(self.n_grid):
                # Adjust x to be at the current grid point
                x[0, self.x_axis_index] = x_vals[i]
                x[0, self.y_axis_index] = y_vals[j]

                # Get the value of the CLF
                V = controller_under_test.V(x)

                # Get the goal, safe, or unsafe classification
                is_goal = controller_under_test.dynamics_model.goal_mask(x).all()
                is_safe = controller_under_test.dynamics_model.safe_mask(x).all()
                is_unsafe = controller_under_test.dynamics_model.unsafe_mask(x).all()

                # Get the QP relaxation
                _, r = controller_under_test.solve_CLF_QP(x)
                relaxation = r.max()

                # Get the linearized CLF value
                P = controller_under_test.dynamics_model.P.type_as(x)
                x0 = controller_under_test.dynamics_model.goal_point.type_as(x)
                P = P.reshape(
                    1,
                    controller_under_test.dynamics_model.n_dims,
                    controller_under_test.dynamics_model.n_dims,
                )
                V_nominal = 0.5 * F.bilinear(x - x0, x - x0, P).squeeze()

                # Store the results
                results.append(
                    {
                        self.x_axis_label: x_vals[i].cpu().numpy().item(),
                        self.y_axis_label: y_vals[j].cpu().numpy().item(),
                        "V": V.cpu().numpy().item(),
                        "QP relaxation": relaxation.cpu().numpy().item(),
                        "Goal region": is_goal.cpu().numpy().item(),
                        "Safe region": is_safe.cpu().numpy().item(),
                        "Unsafe region": is_unsafe.cpu().numpy().item(),
                        "Linearized V": V_nominal.cpu().numpy().item(),
                    }
                )

        return pd.DataFrame(results)

    def plot(
        self,
        controller_under_test: "Controller",
        results_df: pd.DataFrame,
        display_plots: bool = False,
    ) -> List[Tuple[str, figure]]:
        """
        Plot the results, and return the plot handles. Optionally
        display the plots.

        args:
            controller_under_test: the controller with which to run the experiment
            display_plots: defaults to False. If True, display the plots (blocks until
                           the user responds).
        returns: a list of tuples containing the name of each figure and the figure
                 object.
        """
        # Set the color scheme
        sns.set_theme(context="talk", style="white")

        # Plot a contour of V
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(12, 8)

        contours = ax.tricontourf(
            results_df[self.x_axis_label],
            results_df[self.y_axis_label],
            results_df["V"],
            cmap=sns.color_palette("rocket", as_cmap=True),
            levels=20,
        )
        plt.colorbar(contours, ax=ax, orientation="vertical")

        # Plot the linearized CLF
        ax.tricontour(
            results_df[self.x_axis_label],
            results_df[self.y_axis_label],
            results_df["Linearized V"],
            cmap=sns.color_palette("winter", as_cmap=True),
            levels=[0.1],
            linestyles="--",
        )
        ax.tricontour(
            results_df[self.x_axis_label],
            results_df[self.y_axis_label],
            results_df["V"],
            cmap=sns.color_palette("spring", as_cmap=True),
            levels=[0.1],
            linestyles="--",
        )

        # Also overlay the relaxation region
        if results_df["QP relaxation"].max() > 1e-5:
            ax.plot(
                [], [], c=(1.0, 1.0, 1.0, 0.3), label="Certificate Conditions Violated"
            )
            contours = ax.tricontourf(
                results_df[self.x_axis_label],
                results_df[self.y_axis_label],
                results_df["QP relaxation"],
                colors=[(1.0, 1.0, 1.0, 0.3)],
                levels=[1e-5, 1000],
            )

        # And the safe/unsafe regions (if specified)
        if self.plot_unsafe_region:
            ax.plot([], [], c="green", label="Safe Region")
            ax.tricontour(
                results_df[self.x_axis_label],
                results_df[self.y_axis_label],
                results_df["Safe region"],
                colors=["green"],
                levels=[0.5],
            )
            ax.plot([], [], c="magenta", label="Unsafe Region")
            ax.tricontour(
                results_df[self.x_axis_label],
                results_df[self.y_axis_label],
                results_df["Unsafe region"],
                colors=["magenta"],
                levels=[0.5],
            )

            ax.plot([], [], c="blue", label="V(x) = c")
            if hasattr(controller_under_test, "safe_level"):
                ax.tricontour(
                    results_df[self.x_axis_label],
                    results_df[self.y_axis_label],
                    results_df["V"],
                    colors=["blue"],
                    levels=[controller_under_test.safe_level],  # type: ignore
                )
            else:
                ax.tricontour(
                    results_df[self.x_axis_label],
                    results_df[self.y_axis_label],
                    results_df["V"],
                    colors=["blue"],
                    levels=[0.0],
                )

        # Make the legend
        # ax.legend(
        #     bbox_to_anchor=(0, 1.02, 1, 0.2),
        #     loc="lower left",
        #     mode="expand",
        #     borderaxespad=0,
        #     ncol=4,
        # )
        ax.set_xlabel(self.x_axis_label)
        ax.set_ylabel(self.y_axis_label)

        fig_handle = ("V Contour", fig)

        if display_plots:
            plt.show()
            return []
        else:
            return [fig_handle]
