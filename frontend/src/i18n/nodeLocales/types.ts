export interface NodeTranslation {
  description?: string;
  params?: Record<string, string>;
}

export type NodeTranslations = Record<string, NodeTranslation>;
