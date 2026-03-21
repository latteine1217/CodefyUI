import { create } from 'zustand';

interface UIState {
  tooltipsEnabled: boolean;
  toggleTooltips: () => void;
}

const STORAGE_KEY = 'codefyui-tooltips';

export const useUIStore = create<UIState>((set) => ({
  tooltipsEnabled: localStorage.getItem(STORAGE_KEY) !== 'false',
  toggleTooltips: () =>
    set((state) => {
      const next = !state.tooltipsEnabled;
      localStorage.setItem(STORAGE_KEY, String(next));
      return { tooltipsEnabled: next };
    }),
}));
