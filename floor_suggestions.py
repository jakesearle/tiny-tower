import os

from tqdm import tqdm

all_floors = 159


class Tower:

    def __init__(self, gts=0, goal=50, has_shared_living=False):
        self.gts = gts
        self.goal = goal
        self.has_shared_living = has_shared_living

        self.floors_to_build = self.goal - 2

        self.n_bitizens = 0
        self.n_jobs = 0
        self.gt_floor_levels = None
        self.floor_types = None

        self.floor_colors = [f"Red", f"Purple", f"Yellow", f"Blue", f"Green"]

    def print_all_floors(self):
        self.calc_gt_floor_levels()
        self.calc_floor_types()
        shared_living_text = "Shared Living Upgrade" if self.has_shared_living else "No Shared Living Upgrade"
        gt_desc_text = 'Golden Tickets' if self.gts != 1 else 'Golden Ticket'
        gt_subdir_min = ((self.gts - 1) // 50) * 50 + 1
        gt_subdir_max = gt_subdir_min + 49
        gt_text = f'{self.gts:0{len(str(gt_subdir_max))}} {gt_desc_text}'
        gt_subdir_text = f'{gt_subdir_min:03}-{gt_subdir_max} Golden Tickets'

        floor_text = f'{self.goal:03} Floors'
        filepath = f'output/{shared_living_text}/{floor_text}/{gt_subdir_text}/{gt_text}.txt'

        max_floor = len(str(self.goal))
        max_type = max([len(t) for t in self.floor_types])
        lines = [f'{i + 1:>{max_floor}}: {f_type:<{max_type}} (lvl. {gt_lvl})\n' for i, (gt_lvl, f_type) in
                 enumerate(zip(self.gt_floor_levels, self.floor_types))]
        # Write to file:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w+") as outfile:
            outfile.writelines(lines)

    def calc_gt_floor_levels(self):
        if self.gt_floor_levels is not None:
            return
        remaining_gt = self.gts
        floor_levels = [0] * self.floors_to_build
        for level in range(1, 3 + 1):
            for i, f in enumerate(floor_levels):
                if remaining_gt >= level:
                    floor_levels[i] += 1
                    remaining_gt -= level
        self.gt_floor_levels = [0, 0] + floor_levels

    def calc_floor_types(self):
        if self.floor_types is not None:
            return
        self.floor_types = []
        for i, gt_lvl in enumerate(self.gt_floor_levels):
            if i < 4:
                desc = self.add_predetermined_floor(i + 1, gt_lvl)
            elif self.n_bitizens - self.n_jobs >= 3:
                # Build more business floors
                desc = f'{self.floor_colors[self.get_n_businesses() % len(self.floor_colors)]}'
                self.add_business()
            else:
                # Build more residential floors
                self.add_residential(gt_lvl)
                desc = f'Residential'
            self.floor_types.append(desc)

    def add_predetermined_floor(self, i, gt_lvl):
        if i == 1:
            return "Lobby"
        elif i == 2:
            return "Legendary Lounge"
        elif i == 3:
            self.add_business()
            return f"Soda Brewery"
        elif i == 4:
            self.add_residential(gt_lvl)
            return "Residential"

    def add_business(self):
        self.n_jobs += 3

    def add_residential(self, gt_lvl):
        self.n_bitizens += 5
        if self.has_shared_living:
            self.n_bitizens += 1
        if gt_lvl > 1:
            self.n_bitizens += 1

    def get_n_businesses(self):
        return self.n_jobs // 3


def generate_all():
    for extra_person in [True, False]:
        cache = set()
        for gts in tqdm(range(1, 1800 + 1)):
            for goals in range(50, 300 + 1, 50):
                t = Tower(gts=gts, goal=goals, has_shared_living=extra_person)
                t.calc_gt_floor_levels()
                # Include the lowest level in each folder
                if gts >= goals * 6 or (tuple(t.gt_floor_levels) in cache and (gts - 1) % 50 != 0):
                    continue
                cache.add(tuple(t.gt_floor_levels))
                t.print_all_floors()


def main():
    generate_all()


if __name__ == '__main__':
    main()
