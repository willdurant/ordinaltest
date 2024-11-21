from typing import Collection
from scipy import stats
from itertools import combinations


class OrdinalTest:
    def __init__(self) -> None:
        pass

    def add_data(self, groups: Collection[str], values: Collection[int]):
        """Stores the ordinal data.

        Args:
            groups (Collection[str]): States what group each value from the ordinal variable corresponds to.
            values (Collection[int]): States all the observed values from the ordinal variable.
        """
        self.data = {"groups": groups, "values": values}
        self.groups = list(set(groups))

    def get_ranks(self) -> list[int]:
        """Returns the rank of each observed ordinal value, conserving the order of the values.

        Returns:
            list[int]: The rank of each observed value.
        """
        ordinal_values = self.data["values"]
        indexed_numbers = list(enumerate(ordinal_values))
        sorted_pairs = sorted(indexed_numbers, key=lambda x: x[1])

        ranks = {}
        i = 0
        while i < len(sorted_pairs):
            current_value = sorted_pairs[i][1]

            # Find all indices with the same value
            same_value_indices = []
            j = i
            while j < len(sorted_pairs) and sorted_pairs[j][1] == current_value:
                same_value_indices.append(sorted_pairs[j][0])
                j += 1

            # Calculate average rank for ties
            avg_rank = sum(range(i + 1, j + 1)) / len(same_value_indices)

            # Assign the average rank to all tied values
            for idx in same_value_indices:
                ranks[idx] = avg_rank

            i = j

        return [ranks[i] for i in range(len(ordinal_values))]

    def get_group_indices(self) -> dict[str]:
        """Returns a dictionary of indices where each group is represented in the data.

        Returns:
            dict[str]: The dictionary of indices for each group
        """
        group_values = self.data["groups"]
        return {
            group: (i for i, value in enumerate(group_values) if value == group)
            for group in self.groups
        }

    def get_group_counts(self, group_indices: dict[str]) -> dict[str]:
        """Returns the number of observations for each group.

        Args:
            group_indices (dict[str]): The indices of the values for each group.

        Returns:
            dict[str]: The count for each group.
        """
        return {group: len(group_indices[group]) for group in self.groups}

    def get_group_means(
        self,
        group_indices: dict[str],
        group_counts: dict[str],
        value_ranks: Collection[int],
    ) -> dict[str]:
        """Gets the mean rank for each group.

        Args:
            group_indices (dict[str]): The indices for the values for each group.
            value_ranks (Collection[int]): All the ranks for each observed value.

        Returns:
            dict[str]: The mean rank for each group.
        """
        return {
            group: sum(
                [
                    value
                    for i, value in enumerate(value_ranks)
                    if i in group_indices[group]
                ]
            )
            / group_counts[group]
            for group in self.groups
        }

    def calculate_kw_h(
        self, n_cases, group_means, group_counts, expected_rank, variance
    ):
        a = (n_cases - 1) / 12
        b = sum(
            (
                (group_counts[group] * (group_means[group] - expected_rank) ** 2)
                / variance
                for group in self.groups
            )
        )
        return a * b

    def kruskal_wallis(self, alpha):
        if not hasattr(self, "data"):
            raise ValueError("No data provided")

        self.alpha = alpha
        self.group_indices = self.get_group_indices()
        self.value_ranks = self.get_ranks()

        self.group_counts = self.get_group_counts(self.group_indices)
        self.group_means = self.get_group_means(
            self.group_indices, self.group_counts, self.value_ranks
        )
        n_cases = len(self.data["values"])
        expected_rank = (n_cases + 1) / 2
        variance = ((n_cases**2) - 1) / 12

        self.degf = len(self.groups) - 1
        self.h_value = self.calculate_kw_h(
            n_cases, self.group_means, self.group_counts, expected_rank, variance
        )
        self.critical_value = stats.chi2.ppf(1 - self.alpha, self.degf)
        self.p_value = 1 - stats.chi2.cdf(self.h_value, self.degf)
        self.outcome = (
            "Null hypothesis rejected"
            if self.h_value >= self.critical_value
            else "Null hypothesis not rejected"
        )

        print(
            {
                "alpha": self.alpha,
                "degrees of freedom": self.degf,
                "h value": self.h_value,
                "critical chi-square value": self.critical_value,
                "p value": self.p_value,
                "outcome": self.outcome,
            }
        )

    def conover_iman(self):
        if not hasattr(self, "h_value"):
            raise ValueError(
                "Must have rejected null hypothesis from Kruskal-Wallis test to conduct Conover-Iman test."
            )

        if self.outcome == "Null hypothesis not rejected":
            raise ValueError(
                "Must have rejected null hypothesis from Kruskal-Wallis test to conduct Conover-Iman test."
            )

        k = len(self.groups)
        R = sum([value**2 for value in self.data["values"]])

        self.conover_results = dict()
        for Ri, Rj in list(combinations(self.groups, 2)):
            n_i = len(self.group_indices[Ri])
            n_j = len(self.group_indices[Rj])
            n = n_i + n_j
            diff = abs(self.group_means[Ri] - self.group_means[Rj])
            s2 = 1 / (n - 1) * (R - ((n * (n + 1) ** 2) / 4))
            se = s2 * ((n - 1 - self.h_value) / (n - k)) * ((1 / n_i) + (1 / n_j))
            t_value = diff / se

            degf = n - k
            critical_value = stats.t.ppf(1 - self.alpha, degf)
            p_value = 1 - stats.t.cdf(t_value, degf)
            outcome = (
                "Null hypothesis rejected"
                if t_value >= critical_value
                else "Null hypothesis not rejected"
            )

            self.conover_results.setdefault("Ri", []).append(Ri)
            self.conover_results.setdefault("Rj", []).append(Rj)
            self.conover_results.setdefault("degrees of freedom", []).append(degf)
            self.conover_results.setdefault("t value", []).append(t_value)
            self.conover_results.setdefault("critical value", []).append(critical_value)
            self.conover_results.setdefault("p value", []).append(p_value)
            self.conover_results.setdefault("outcome", []).append(outcome)

        print(self.conover_results)
