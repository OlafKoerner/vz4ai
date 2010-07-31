<?php

namespace Volkszaehler\Model\Proxies;

/**
 * THIS CLASS WAS GENERATED BY THE DOCTRINE ORM. DO NOT EDIT THIS FILE.
 */
class VolkszaehlerModelTokenProxy extends \Volkszaehler\Model\Token implements \Doctrine\ORM\Proxy\Proxy
{
    private $_entityPersister;
    private $_identifier;
    public $__isInitialized__ = false;
    public function __construct($entityPersister, $identifier)
    {
        $this->_entityPersister = $entityPersister;
        $this->_identifier = $identifier;
    }
    private function _load()
    {
        if (!$this->__isInitialized__ && $this->_entityPersister) {
            $this->__isInitialized__ = true;
            if ($this->_entityPersister->load($this->_identifier, $this) === null) {
                throw new \Doctrine\ORM\EntityNotFoundException();
            }
            unset($this->_entityPersister, $this->_identifier);
        }
    }

    
    public function __toString()
    {
        $this->_load();
        return parent::__toString();
    }

    public function getToken()
    {
        $this->_load();
        return parent::getToken();
    }


    public function __sleep()
    {
        return array('__isInitialized__', 'id', 'token', 'valid', 'entity');
    }
}