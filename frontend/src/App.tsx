import { ReactFlowProvider } from '@xyflow/react';
import { Toolbar } from './components/Toolbar/Toolbar';
import { TabBar } from './components/TabBar/TabBar';
import { NodePalette } from './components/Sidebar/NodePalette';
import { FlowCanvas } from './components/Canvas/FlowCanvas';
import { NodeConfigPanel } from './components/ConfigPanel/NodeConfigPanel';
import { ResultsPanel } from './components/ResultsPanel/ResultsPanel';
import { PresetConfigModal } from './components/PresetModal/PresetConfigModal';
import { useTabStore } from './store/tabStore';

function TabContent({ tabId }: { tabId: string }) {
  const activeTabId = useTabStore((s) => s.activeTabId);
  const isActive = tabId === activeTabId;

  return (
    <div
      style={{
        flex: 1,
        display: isActive ? 'flex' : 'none',
        flexDirection: 'column',
        overflow: 'hidden',
        minHeight: 0,
      }}
    >
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>
        <ReactFlowProvider>
          <NodePalette />
          <div style={{ flex: 1, position: 'relative', minWidth: 0 }}>
            <div style={{ width: '100%', height: '100%' }}>
              <FlowCanvas />
            </div>
          </div>
          <NodeConfigPanel />
        </ReactFlowProvider>
      </div>
      <ResultsPanel />
    </div>
  );
}

function App() {
  const tabs = useTabStore((s) => s.tabs);

  return (
    <div
      style={{
        width: '100vw',
        height: '100vh',
        background: '#121212',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        color: '#eeeeee',
        fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
      }}
    >
      <Toolbar />
      <TabBar />
      {tabs.map((tab) => (
        <TabContent key={tab.id} tabId={tab.id} />
      ))}
      <PresetConfigModal />
    </div>
  );
}

export default App;
