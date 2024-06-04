import math
import random
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# NODO: TP upgrade calculation -- It takes so long this doesn't matter
# DONE: Hollywood bonus
#   - TP calculation
#   - rebuild calculations
# DONE: Dice rolls -- Did an approximation

# GT_PER_DAY = 0
# MAX_TP_PER_DAY = 6
# AVE_TP_PER_DAY = 4

LB_COUNTS = {
    "Animated Movies": 6,
    "Fantasy Movies": 6,
    "Animated Series": 7,
    "Sci-fi": 8,
    "Video Games": 8,
    "Music Icons": 6,
    "American Sports": 7,
    "Acting Icons": 7,
    "Greek Gods": 7,
    "Pop Culture Pantheon": 7
}
TOTAL_LBS = sum(LB_COUNTS.values())

MAX_DAYS = 10_000

# Duplicate counts per level
LVL_1 = 1  # 1
LVL_2 = 1  # 2
LVL_3 = 3  # 5
LVL_4 = 4  # 9
LVL_5 = 6  # 15
MAX_LB = 15
MAX_LVL = 5

# Base Costs
BRONZE_BASE_COST = 100
SILVER_BASE_COST = 50
GOLD_BASE_COST = 5
BRONZE_STEP = 20
SILVER_STEP = 10
GOLD_STEP = 5
COST_CAP = 500

REFUND_AMT = 500

# Case Odds
FREE_CASE_ODDS = .01
AD_CASE_ODDS = .02

# Estimated LBs earned from events
# I don't want to calculate this out, but this is based on how I play the events
# I've been playing for 42 days since the legendary bitizens have been released
# In that time I've earned 35 LBs total
# According to my simulation, on average I would have earned 6.8 LBs (rounded to 7) so far from keys and free boxes
EVENT_LBS_PER_DAY = (35 - 7) / 42


class User:
    def __init__(self):
        self.legendaries = {k: [0 for _ in range(v)] for k, v in LB_COUNTS.items()}

        self.bronze_count = BRONZE_BASE_COST
        self.silver_count = SILVER_BASE_COST
        self.gold_count = GOLD_BASE_COST

        self.bronze_cost = BRONZE_BASE_COST
        self.silver_cost = SILVER_BASE_COST
        self.gold_cost = GOLD_BASE_COST

        self.has_hollywood_7 = True
        self.curr_day = 0
        self.day_log = []
        self.total_event_lbs = 0.0

    def do_day(self):
        # Event bitizens
        self.get_event_bitizens()
        # Redeem cases
        self.redeem_cases()
        # Collect keys
        self.collect_keys()
        # Spend keys
        self.spend_keys()
        # Log day
        self.log_day()
        # End day
        self.curr_day += 1

    def is_finished(self):
        return all(all(i == MAX_LB for i in v) for v in self.legendaries.values())

    def redeem_cases(self):
        self.redeem_case(FREE_CASE_ODDS)
        self.redeem_case(FREE_CASE_ODDS)
        self.redeem_case(FREE_CASE_ODDS)

        self.redeem_case(AD_CASE_ODDS)
        self.redeem_case(AD_CASE_ODDS)
        self.redeem_case(AD_CASE_ODDS)

    def redeem_case(self, odds):
        if random.random() <= odds:
            self.get_rando()

    def get_rando(self):
        self.add_lb(index=random.randrange(TOTAL_LBS))

    def add_lb(self, index=None, ki=None):
        if ki is None:
            k, i = self.get_key_and_index(index)
        else:
            k, i = ki
        if self.legendaries[k][i] == MAX_LB:
            self.do_refund()
            return
        self.legendaries[k][i] += 1

    def get_key_and_index(self, index):
        for k, v in self.legendaries.items():
            if index < len(v):
                return k, index
            index -= len(v)

    def do_refund(self):
        self.bronze_count += REFUND_AMT

    def collect_keys(self):
        if self.has_hollywood_7 and self.curr_day % 7 == 0:
            self.silver_count += 2
        for k, v in self.legendaries.items():
            for num_lb in v:
                lvl = get_lvl(num_lb)
                match lvl:
                    case 3:
                        self.get_bronze()
                    case 4:
                        self.get_silver()
                    case 5:
                        self.get_gold()

    def get_bronze(self):
        self.bronze_count += 1

    def get_silver(self):
        self.silver_count += 1

    def get_gold(self):
        self.gold_count += 1

    def spend_keys(self):
        self.spend_bronze()
        self.spend_silver()
        self.spend_gold()
        if not self.is_finished() and (self.bronze_choice() or self.silver_choice() or self.gold_choice()):
            self.spend_keys()

    def spend_bronze(self):
        if not self.bronze_choice():
            return
        # "Pay"
        self.bronze_count -= self.bronze_cost
        self.bronze_cost = min(COST_CAP, self.bronze_cost + BRONZE_STEP)
        # "Choose"
        self.get_rando()

    def bronze_choice(self):
        return self.bronze_count >= self.bronze_cost

    def spend_silver(self):
        choice = self.silver_choice()
        if not choice:
            return
        # "Pay"
        self.silver_count -= self.silver_cost
        self.silver_cost = min(COST_CAP, self.silver_cost + SILVER_STEP)
        # "Choose"
        index = self.random_index(choice)
        self.add_lb(ki=(choice, index))

    def silver_choice(self):
        # Spend silver ASAP
        if self.silver_count < self.silver_cost:
            return False
        # Return the first group with the highest level LB
        key = ''
        mx = -1
        for k, v in self.legendaries.items():
            for lb in v:
                if lb > mx:
                    mx = lb
                    key = k
        return key

    def spend_gold(self):
        choice = self.gold_choice()
        if not choice:
            return
        self.gold_count -= self.gold_cost
        self.gold_cost = min(COST_CAP, self.gold_cost + GOLD_STEP)
        self.add_lb(ki=choice)

    def gold_choice(self):
        # Get the first bitizen that can be leveled up to the highest level
        # Spend gold ASAP
        if self.gold_count < self.gold_cost:
            return False
        best = None
        mx_lvl = -1
        # Close to upgrading
        for k, v in self.legendaries.items():
            for i, lb in enumerate(v):
                # Close to upgrading
                if get_lvl(lb) < get_lvl(lb + 1) and get_lvl(lb + 1) > mx_lvl:
                    best = (k, i)
                    mx_lvl = get_lvl(lb + 1)
        # I'll just save my gold
        if self.gold_count < self.gold_cost * 2:
            return best
        # Just spend it
        max_num = -1
        for k, v in self.legendaries.items():
            for i, lb in enumerate(v):
                # Close to upgrading
                if lb > max_num:
                    best = (k, i)
                    max_num = lb
        return best

    def random_index(self, group):
        return random.randrange(len(self.legendaries[group]))

    def sum(self):
        return sum(sum(v) for v in self.legendaries.values())

    def log_day(self):
        self.day_log.append((self.curr_day, self.sum()))

    def graph(self):
        print(len(self.day_log))
        x, y = zip(*self.day_log)
        sns.lineplot(x=x, y=y)
        plt.show()

    def get_event_bitizens(self):
        self.total_event_lbs += EVENT_LBS_PER_DAY
        if self.total_event_lbs < 1.0:
            return
        whole_people = math.floor(self.total_event_lbs)
        for _ in range(whole_people):
            self.get_rando()
        self.total_event_lbs -= whole_people


def get_lvl(n):
    if n < 1:
        return 0
    elif n < 2:
        return 1
    elif n < 5:
        return 2
    elif n < 9:
        return 3
    elif n < 15:
        return 4
    return 5


def average_time():
    def sim():
        user = User()
        for i in range(MAX_DAYS):
            user.do_day()
            if user.is_finished():
                return i
        return MAX_DAYS

    n = 10_000
    times = [sim() for _ in tqdm(range(n))]
    avg = round(sum(times) / len(times))
    print(f'{n} simulations, {avg // 365} years {avg % 365} days')


def composite_graph():
    def sim():
        u = User()
        for _ in range(MAX_DAYS):
            u.do_day()
            if u.is_finished():
                return u.day_log

    n = 100
    logs = [sim() for _ in tqdm(range(n))]
    xs = [x for log in logs for (x, _) in log]
    ys = [y for log in logs for (_, y) in log]
    # Just every tenth day
    sample = 10
    xs = [x for i, x in enumerate(xs) if i % sample == 0]
    ys = [y for i, y in enumerate(ys) if i % sample == 0]

    sns.lineplot(x=xs, y=ys, errorbar="se", sort=True)
    plt.title(f'Legendary Bitizens Over Time (n={n})')
    plt.xlabel('Days')
    plt.ylabel('Total Legendary Bitizens Collected')
    plt.show()


def main():
    composite_graph()


if __name__ == '__main__':
    main()
