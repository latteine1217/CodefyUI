import { NodeContextMenu } from '../ContextMenu/NodeContextMenu';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';

interface PaneContextMenuProps {
  screen: { x: number; y: number };
  flow: { x: number; y: number };
  onClose: () => void;
}

export function PaneContextMenu({ screen, flow, onClose }: PaneContextMenuProps) {
  const addNote = useTabStore((s) => s.addNote);
  const { t } = useI18n();

  const items = [
    {
      label: t('contextMenu.addTextNote'),
      action: () => addNote('text', flow),
    },
    {
      label: t('contextMenu.addImageNote'),
      action: () => addNote('image', flow),
    },
  ];

  return (
    <NodeContextMenu
      position={{ nodeId: '', x: screen.x, y: screen.y }}
      items={items}
      onClose={onClose}
    />
  );
}
