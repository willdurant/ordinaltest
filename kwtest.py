from typing import Collection
from scipy import stats


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
        if not self.data:
            raise ValueError("No data provided.")

        group_indices = self.get_group_indices()
        value_ranks = self.get_ranks()

        group_counts = self.get_group_counts(group_indices)
        group_means = self.get_group_means(group_indices, group_counts, value_ranks)
        n_cases = len(self.data["values"])
        expected_rank = (n_cases + 1) / 2
        variance = ((n_cases**2) - 1) / 12

        degf = len(self.groups) - 1
        self.h_value = self.calculate_kw_h(
            n_cases, group_means, group_counts, expected_rank, variance
        )
        self.critical_value = stats.chi2.ppf(1 - alpha, degf)
        if self.h_value < self.critical_value:
            self.outcome = "Null hypothesis not rejected"
        else:
            self.outcome = "Null hypothesis rejected"

        print(
            {
                "alpha": alpha,
                "h value": self.h_value,
                "critical chi-square value": self.critical_value,
                "outcome": self.outcome,
            }
        )
