import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ContextMenu } from './ContextMenu';

describe('ContextMenu', () => {
  it('renders items at the given position', () => {
    const onSelect = vi.fn();
    render(
      <ContextMenu
        x={100}
        y={200}
        items={[{ id: 'foo', label: 'Foo Action' }]}
        onSelect={onSelect}
        onClose={() => {}}
      />
    );
    const item = screen.getByText('Foo Action');
    expect(item).toBeInTheDocument();
  });

  it('fires onSelect with item id when clicked', () => {
    const onSelect = vi.fn();
    const onClose = vi.fn();
    render(
      <ContextMenu
        x={0}
        y={0}
        items={[{ id: 'foo', label: 'Foo' }, { id: 'bar', label: 'Bar' }]}
        onSelect={onSelect}
        onClose={onClose}
      />
    );
    fireEvent.click(screen.getByText('Bar'));
    expect(onSelect).toHaveBeenCalledWith('bar');
    expect(onClose).toHaveBeenCalled();
  });

  it('closes on outside click', () => {
    const onClose = vi.fn();
    render(
      <ContextMenu
        x={0}
        y={0}
        items={[{ id: 'foo', label: 'Foo' }]}
        onSelect={() => {}}
        onClose={onClose}
      />
    );
    fireEvent.mouseDown(document.body);
    expect(onClose).toHaveBeenCalled();
  });
});
